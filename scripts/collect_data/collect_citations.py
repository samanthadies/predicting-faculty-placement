"""
collect_citations.py

End-to-end pipeline to:
  1. Discover OpenAlex author candidates for each (author, university) in faculty_dec.csv.
  2. Resolve candidates using DBLP titles and fetch citation counts_by_year from OpenAlex.
  3. Convert per-year counts to cumulative citations (2010–2020) and merge into faculty.csv.

Stage 1 output:
  - ../data/hiring/openalex_author_candidates.csv

Stage 2 output:
  - ../data/hiring/openalex_author_candidates_resolved.csv
  - ../data/hiring/openalex_author_unresolved.csv   (authors with no resolved OpenAlex ID)

Stage 3 output:
  - ../data/hiring/faculty.csv

Assumptions:
  - faculty.csv has columns: ['author', 'university', ...]
  - DBLP file full_by_author.csv has columns: ['author', 'title', ...]

12/11/2025 - SD
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from tqdm.auto import tqdm


OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_MAILTO = "dies.s@northeastern.edu"


def rate_limited_get(url, params, base_sleep=0.5, max_retries=5):
    """
    Helper for GET requests with basic rate limiting and retry on 429/5xx.
    Returns parsed JSON or None on failure.

    :param url: OpenAlex url
    :param params: OpenAlex search parameters
    :param base_sleep: sleep between requests
    :param max_retries: number of retries per entry
    :return: OpenAlex data or None
    """

    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, params=params, timeout=30)
            status = resp.status_code

            # Success
            if status == 200:
                time.sleep(base_sleep)
                return resp.json()

            # Too many requests: back off and retry
            if status == 429:
                wait = base_sleep * (2 ** attempt)
                print(f"Got 429 for {resp.url}, sleeping {wait:.1f}s and retrying...")
                time.sleep(wait)
                attempt += 1
                continue

            # Client errors (400, 404, etc.) – don’t retry
            if 400 <= status < 500:
                print(f"Warning: status {status} for URL {resp.url}")
                return None

            # Server errors (5xx): retry with backoff
            if status >= 500:
                wait = base_sleep * (2 ** attempt)
                print(f"Server error {status} for {resp.url}, sleeping {wait:.1f}s and retrying...")
                time.sleep(wait)
                attempt += 1
                continue

        except Exception as e:
            print(f"Error fetching {url} with params {params}: {e}")
            attempt += 1
            time.sleep(base_sleep * (2 ** attempt))

    print(f"Giving up on {url} after {max_retries} attempts.")
    return None


def normalize(s):
    """
    Lowercase, strip, and collapse whitespace.

    :param s: String to normalize
    :return: the normalized string
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def name_tokens(s):
    """
    Tokenize a name on whitespace/dashes and strip simple punctuation.
    Returns a list of tokens (e.g., ['samantha', 'dies']).

    :param s: name to tokenize
    :return: tokenized name
    """
    s = normalize(s)
    if not s:
        return []
    tokens = re.split(r"[ \t\-]+", s)
    return [t.strip(".,()") for t in tokens if t.strip(".,()")]


def names_roughly_match(target_name, candidate_name):
    """
    Heuristic name match: Last names must match exactly;
    First names either match exactly OR first letters match (Sam vs Samantha).

    :param target_name: target name from faculty.csv
    :param candidate_name: candidate name from OpenAlex
    :return: True/False depending on the match
    """

    t_tokens = name_tokens(target_name)
    c_tokens = name_tokens(candidate_name)

    if len(t_tokens) < 2 or len(c_tokens) < 2:
        return False

    t_first, t_last = t_tokens[0], t_tokens[-1]
    c_first, c_last = c_tokens[0], c_tokens[-1]

    if t_last != c_last:
        return False

    # exact first name match OR first-letter match
    if t_first == c_first:
        return True
    if t_first[0] == c_first[0]:
        return True

    return False


def institution_roughly_matches(university, candidate):
    """
     Heuristic institution match: Look at last_known_institution.display_name or
     first element of last_known_institutions. Check simple substring overlap
     with the faculty 'university' string.

    :param university: university from faculty.csv
    :param candidate: candidate from OpenAlex
    :return: True/False depending on match
    """

    uni_norm = normalize(university)

    # Try singular
    lki = candidate.get("last_known_institution")
    if not lki:
        # Try plural
        lkis = candidate.get("last_known_institutions") or []
        lki = lkis[0] if lkis else None

    if not lki:
        return False

    inst_name = normalize(lki.get("display_name", ""))
    if not inst_name or not uni_norm:
        return False

    # crude overlap: one name contains the other
    return uni_norm in inst_name or inst_name in uni_norm


def filter_candidates_for_author(author_name, university, candidates, max_keep=3):
    """
    Apply simple heuristics to reduce the candidate list:
      1. Keep candidates with good name AND institution match.
      2. If that’s empty, keep candidates with good name match.
      3. If still empty, keep just the top-1 candidate.
    Then truncate to at most `max_keep` candidates.

    :param author_name: name from faculty.csv
    :param university: university from faculty.csv
    :param candidates: OpenAlex results
    :param max_keep: max candidates to keep
    :return: list of filtered candidates
    """

    if not candidates:
        return []

    strong = []
    name_only = []
    for cand in candidates:
        cname = cand.get("display_name", "") or ""
        name_ok = names_roughly_match(author_name, cname)
        inst_ok = institution_roughly_matches(university, cand)

        if name_ok and inst_ok:
            strong.append(cand)
        elif name_ok:
            name_only.append(cand)

    if strong:
        filtered = strong
    elif name_only:
        filtered = name_only
    else:
        # nothing matches heuristics; fall back to top-1 OpenAlex guess
        filtered = [candidates[0]]

    return filtered[:max_keep]


def search_openalex_author(author_name):
    """
    Search OpenAlex for an author given a full name.

    :param author_name: name from faculty.csv to search
    :return: filtered results
    """

    url = f"{OPENALEX_BASE}/authors"

    name = (author_name or "").strip()
    if not name:
        return []

    params = {
        "search": name,
        "per-page": 5,  # a few candidates for manual disambiguation
    }
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    data = rate_limited_get(url, params)
    if data is None or "results" not in data:
        return []

    results = data["results"] or []

    # Optional: quick debug to sanity-check in the console
    if results:
        top = results[0]
        top_name = top.get("display_name")
        print(f"  -> got {len(results)} candidates, top='{top_name}'")
    else:
        print("  -> no candidates found")

    return results


def flatten_author_candidate(author, university, candidate, rank):
    """
    Turn a single OpenAlex author candidate into a flat dict suitable for a CSV row.
    Only keep lightweight fields needed for later resolution.

    :param author: author name from faculty.csv
    :param university: university from faculty.csv
    :param candidate: candidates from OpenAlex
    :param rank: the rank of the candidate
    :return: final rows for initial round of API requests
    """
    oa_id = candidate.get("id", "")
    display_name = candidate.get("display_name", "")
    orcid = candidate.get("orcid", "")
    works_count = candidate.get("works_count", None)
    cited_by_count = candidate.get("cited_by_count", None)

    # last known institution (singular)
    lki = candidate.get("last_known_institution") or {}
    lki_id = lki.get("id", "")
    lki_name = lki.get("display_name", "")

    # publication range from summary_stats (if available)
    summary_stats = candidate.get("summary_stats") or {}
    first_year = summary_stats.get("2yr_mean_citedness_start_year", None)
    last_year = summary_stats.get("2yr_mean_citedness_end_year", None)

    # works endpoint (for fetching titles later)
    works_api_url = candidate.get("works_api_url", "")

    return {
        "author": author,
        "university": university,
        "candidate_rank": rank,
        "openalex_id": oa_id,
        "openalex_display_name": display_name,
        "openalex_orcid": orcid,
        "openalex_works_count": works_count,
        "openalex_cited_by_count": cited_by_count,
        "openalex_last_known_institution_id": lki_id,
        "openalex_last_known_institution_name": lki_name,
        "openalex_first_year_est": first_year,
        "openalex_last_year_est": last_year,
        "openalex_works_api_url": works_api_url,
    }


def build_author_candidate_table(faculty_csv, output_csv, max_authors=None):
    """
    For each unique (author, university) pair in faculty_dec.csv, query OpenAlex for
    candidate author matches and save the candidates to a CSV for semi-manual validation.

    :param faculty_csv: faculty csv
    :param output_csv: csv path to save to
    :param max_authors: max candidates per author
    :return: None
    """

    print(f"Loading faculty data from: {faculty_csv}")
    df = pd.read_csv(faculty_csv)

    if "author" not in df.columns or "university" not in df.columns:
        raise ValueError("Expected 'author' and 'university' columns in faculty_dec.csv")

    # Unique author-university pairs for search
    unique_pairs = df[["author", "university"]].drop_duplicates()
    print(f"Found {len(unique_pairs)} unique (author, university) pairs.")

    if max_authors is not None:
        unique_pairs = unique_pairs.head(max_authors)
        print(f"Limiting to first {len(unique_pairs)} pairs for prototyping.")

    rows = []

    for i, (_, row) in enumerate(unique_pairs.iterrows(), start=1):
        author_name = str(row["author"]).strip()
        university = str(row["university"]).strip()

        if not author_name:
            continue

        print(f"[{i}/{len(unique_pairs)}] Searching OpenAlex for '{author_name}' ({university})")
        candidates = search_openalex_author(author_name)

        if not candidates:
            # Explicit "no candidates" row
            rows.append(
                {
                    "author": author_name,
                    "university": university,
                    "candidate_rank": None,
                    "openalex_id": None,
                    "openalex_display_name": None,
                    "openalex_orcid": None,
                    "openalex_works_count": None,
                    "openalex_cited_by_count": None,
                    "openalex_last_known_institution_id": None,
                    "openalex_last_known_institution_name": None,
                    "openalex_first_year_est": None,
                    "openalex_last_year_est": None,
                    "openalex_works_api_url": None,
                }
            )
            continue

        # Apply heuristics to reduce candidate list
        filtered_candidates = filter_candidates_for_author(
            author_name=author_name,
            university=university,
            candidates=candidates,
            max_keep=3,  # up to 3 candidates per (author, university)
        )

        for rank, cand in enumerate(filtered_candidates):
            flat = flatten_author_candidate(author_name, university, cand, rank)
            rows.append(flat)

    out_df = pd.DataFrame(rows)
    print(f"Saving {len(out_df)} rows to {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print("Sample authors from candidate table:")
    print(out_df["author"].head())


def normalize_title(title):
    """
    Simple normalization for comparing titles between DBLP and OpenAlex.

    :param title: paper titles to normalize
    :return: normalized titles
    """

    if not isinstance(title, str):
        return ""
    t = title.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)  # remove punctuation-ish characters
    return t.strip()


def load_author_title_sets(dblp_csv):
    """
    Load ../data/coauthorship/dblp_cleaned/full_by_author.csv and
    build a mapping: author -> set of normalized titles.

    :param dblp_csv: dblp csv with papers
    :return: dictionary with papers for each author
    """

    df = pd.read_csv(dblp_csv)
    if "author" not in df.columns or "title" not in df.columns:
        raise ValueError("Expected 'author' and 'title' columns in DBLP CSV")

    df["title_norm"] = df["title"].apply(normalize_title)

    mapping = {}
    grouped = df.groupby("author")["title_norm"]
    for author, titles in grouped:
        mapping[author] = {t for t in titles if t}

    print(f"Loaded DBLP titles for {len(mapping)} authors from {dblp_csv}")
    return mapping


def fetch_openalex_author_titles(works_api_url, max_works=300):
    """
    Given an OpenAlex works_api_url like:
        https://api.openalex.org/works?filter=author.id:A123...
    fetch up to `max_works` works and return a set of normalized titles.

    :param works_api_url: URL from round 1
    :param max_works: max numbers of papers to check
    :return: set of titles
    """

    if not works_api_url or not isinstance(works_api_url, str):
        return set()

    titles = set()
    page_url = works_api_url
    fetched = 0

    while page_url and fetched < max_works:
        # Split url into base + params for rate_limited_get
        if "?" in page_url:
            base, query_str = page_url.split("?", 1)
            params = dict(
                kv.split("=", 1) if "=" in kv else (kv, "")
                for kv in query_str.split("&")
                if kv
            )
        else:
            base = page_url
            params = {}

        if OPENALEX_MAILTO:
            params.setdefault("mailto", OPENALEX_MAILTO)
        params.setdefault("per-page", "200")

        data = rate_limited_get(base, params)
        if data is None:
            break

        results = data.get("results") or []
        for w in results:
            title = w.get("title", "")
            t_norm = normalize_title(title)
            if t_norm:
                titles.add(t_norm)
                fetched += 1
                if fetched >= max_works:
                    break

        # Pagination: OpenAlex uses meta.next as a full URL for the next page
        meta = data.get("meta", {})
        next_url = meta.get("next")
        if not next_url:
            break
        page_url = next_url

    return titles


def fetch_author_counts_by_year(openalex_id):
    """
    Given an OpenAlex author ID like 'https://openalex.org/A5100460246' or 'A5100460246',
    fetch the author record and return the counts_by_year list (or None on failure).

    :param openalex_id: get the author record and return the counts_by_year
    :return: counts_by_year
    """

    if not isinstance(openalex_id, str) or not openalex_id:
        return None

    # Accept either full URL or bare ID
    if openalex_id.startswith("http"):
        author_key = openalex_id.rsplit("/", 1)[-1]
    else:
        author_key = openalex_id

    url = f"{OPENALEX_BASE}/authors/{author_key}"
    params = {}
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    data = rate_limited_get(url, params)
    if data is None:
        return None

    cby = data.get("counts_by_year")
    if not isinstance(cby, list):
        return None
    return cby


def resolve_candidates_with_dblp(candidates_csv, dblp_csv, output_csv, max_works_per_candidate=300):
    """
    Stage 2, part 1 + 2:

      1. For each (author, university) in the candidates table, use DBLP titles to
         try to resolve to a single OpenAlex candidate.
      2. Collapse to one row per (author, university) and attach counts_by_year JSON
         for resolved authors.

    Output:
      - output_csv: one row per (author, university) with OpenAlex fields +
                    dblp_match_count + resolved flag + openalex_counts_by_year_json.

    :param candidates_csv: csv with initial OpenAlex requests
    :param dblp_csv: dblp csv with publications
    :param output_csv: output csv
    :param max_works_per_candidate: max papers to check per author
    :return: None
    """

    # Load candidate table from Stage 1
    df = pd.read_csv(candidates_csv)

    # Load DBLP titles mapping from full_by_author.csv
    author_to_titles = load_author_title_sets(dblp_csv)

    # Prepare new columns
    df["dblp_match_count"] = 0
    df["resolved"] = 0

    # Cache OpenAlex title sets so we don't fetch the same author twice
    works_title_cache = {}

    group_cols = ["author", "university"]
    grouped = df.groupby(group_cols, dropna=False)

    grouped_items = list(grouped)
    total_groups = len(grouped_items)
    print(f"Resolving candidates for {total_groups} (author, university) pairs...")

    if tqdm is not None:
        iterator = tqdm(grouped_items, desc="Step 1/2: DBLP-based resolution", unit="author")
    else:
        iterator = grouped_items

    all_groups = []

    # Step 1: choose best candidate per group based on title overlap
    for (author, university), grp in iterator:
        author = str(author)
        grp = grp.copy()

        dblp_titles = author_to_titles.get(author, set())
        if not dblp_titles:
            # No DBLP info for this author; nothing we can do here
            all_groups.append(grp)
            continue

        best_idx = None
        best_match_count = 0

        for idx, row in grp.iterrows():
            oa_id = row.get("openalex_id")
            works_api_url = row.get("openalex_works_api_url")

            if pd.isna(oa_id) or not isinstance(oa_id, str):
                continue

            # Use cache keyed by openalex_id
            if oa_id in works_title_cache:
                oa_titles = works_title_cache[oa_id]
            else:
                oa_titles = fetch_openalex_author_titles(
                    works_api_url=works_api_url,
                    max_works=max_works_per_candidate,
                )
                works_title_cache[oa_id] = oa_titles

            if not oa_titles:
                continue

            match_count = len(dblp_titles.intersection(oa_titles))
            grp.at[idx, "dblp_match_count"] = match_count

            if match_count > best_match_count:
                best_match_count = match_count
                best_idx = idx

        # Mark best candidate as resolved if we saw any overlap at all
        if best_idx is not None and best_match_count > 0:
            grp.loc[grp.index == best_idx, "resolved"] = 1

        all_groups.append(grp)

    resolved_df = pd.concat(all_groups, ignore_index=True)

    total_rows = len(resolved_df)
    num_resolved_rows = int(resolved_df["resolved"].sum())
    print(f"Total candidate rows (before collapse): {total_rows}")
    print(f"Resolved candidates (dblp_match_count > 0): {num_resolved_rows}")

    # Step 2: collapse to one row per (author, university) and fetch counts_by_year
    collapsed_rows = []

    grouped2 = resolved_df.groupby(group_cols, dropna=False)
    grouped2_items = list(grouped2)
    total_groups2 = len(grouped2_items)

    print("Collapsing to one row per (author, university) and fetching counts_by_year for resolved authors...")

    if tqdm is not None:
        iterator2 = tqdm(grouped2_items, desc="Step 2/2: Collapse + counts_by_year", unit="author")
    else:
        iterator2 = grouped2_items

    for (author, university), grp in iterator2:
        grp = grp.copy()

        # Try to find a resolved candidate
        resolved_rows = grp[grp["resolved"] == 1]
        if not resolved_rows.empty:
            # Take the first resolved row
            row = resolved_rows.iloc[0].copy()
        else:
            # Create an unresolved dummy row based on the first group row
            row = grp.iloc[0].copy()
            # Clear OpenAlex-specific fields
            for col in [
                "candidate_rank",
                "openalex_id",
                "openalex_display_name",
                "openalex_orcid",
                "openalex_works_count",
                "openalex_cited_by_count",
                "openalex_last_known_institution_id",
                "openalex_last_known_institution_name",
                "openalex_first_year_est",
                "openalex_last_year_est",
                "openalex_works_api_url",
            ]:
                if col in row.index:
                    row[col] = None
            row["dblp_match_count"] = 0
            row["resolved"] = 0

        # Fetch counts_by_year for resolved authors
        counts_by_year_json = None
        if int(row.get("resolved", 0)) == 1:
            oa_id = row.get("openalex_id")
            if isinstance(oa_id, str) and oa_id:
                cby = fetch_author_counts_by_year(oa_id)
                if cby is not None:
                    counts_by_year_json = json.dumps(cby)

        row_dict = row.to_dict()
        row_dict["openalex_counts_by_year_json"] = counts_by_year_json
        collapsed_rows.append(row_dict)

    collapsed_df = pd.DataFrame(collapsed_rows)

    num_authors = len(collapsed_df)
    num_resolved_authors = int(collapsed_df["resolved"].sum())

    print(f"Collapsed to {num_authors} (author, university) rows.")
    print(f"Authors with a resolved OpenAlex ID: {num_resolved_authors}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    collapsed_df.to_csv(output_csv, index=False)
    print(f"Saved collapsed resolved mapping to {output_csv}")


def export_unresolved_authors(resolved_csv, unresolved_out):
    """
    Convenience helper:

      - Read openalex_author_candidates_resolved.csv
      - Filter rows with resolved == 0
      - Save to openalex_author_unresolved.csv for manual check

    :param resolved_csv: csv with citations from OpenAlex
    :param unresolved_out: csv with authors who need manual collection
    :return: None
    """

    df = pd.read_csv(resolved_csv)
    df_unresolved = df[df["resolved"] == 0].copy()
    print(df_unresolved.info())
    unresolved_out.parent.mkdir(parents=True, exist_ok=True)
    df_unresolved.to_csv(unresolved_out, index=False)
    print(f"Saved unresolved authors to {unresolved_out}")


def parse_counts_by_year_json(json_str):
    """
    Safely parse the counts_by_year JSON field.

    :param json_str: cited_by_year json to parse
    :return: parsed json
    """

    if pd.isna(json_str) or json_str is None:
        return None

    if isinstance(json_str, list):
        # Already parsed
        return json_str

    if not isinstance(json_str, str):
        return None

    json_str = json_str.strip()
    if not json_str:
        return None

    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None


def compute_cumulative_citations(counts_by_year, start_year=2010, end_year=2020):
    """
    Given counts_by_year from OpenAlex, compute cumulative citations for each year
    from start_year to end_year (inclusive).

    :param counts_by_year: OpenAlex json with citations
    :param start_year: start year
    :param end_year: end year
    :return: citaiton columns for faculty.csv
    """

    cols = {f"citations_{year}": 0.0 for year in range(start_year, end_year + 1)}

    if not counts_by_year:
        return cols

    # Build a simple year -> cited_by_count mapping
    year_to_cites = {}
    for entry in counts_by_year:
        try:
            year = int(entry.get("year"))
        except Exception:
            continue
        cbc = entry.get("cited_by_count", 0) or 0
        try:
            cbc = float(cbc)
        except Exception:
            cbc = 0.0
        year_to_cites[year] = year_to_cites.get(year, 0.0) + cbc

    # Baseline: all citations before the first year of interest
    baseline = sum(c for y, c in year_to_cites.items() if y < start_year)

    cumulative = baseline
    for year in range(start_year, end_year + 1):
        cumulative += year_to_cites.get(year, 0.0)
        cols[f"citations_{year}"] = cumulative

    return cols


def build_citation_table_from_openalex(resolved_csv, start_year=2010, end_year=2020):
    """
    Read openalex_author_candidates_resolved.csv and produce a small dataframe
    with (author, university) + cumulative citation columns for start_year to
    end_year using the counts_by_year JSON field.

    :param resolved_csv: csv with citations from OpenAlex
    :param start_year: start year
    :param end_year: end year
    :return: small dataframe with final citation counts
    """

    df_resolved = pd.read_csv(resolved_csv)

    if "author" not in df_resolved.columns or "university" not in df_resolved.columns:
        raise ValueError("Expected 'author' and 'university' in resolved CSV")

    if "openalex_counts_by_year_json" not in df_resolved.columns:
        raise ValueError(
            "Expected 'openalex_counts_by_year_json' in resolved CSV. "
            "Make sure resolve_candidates_with_dblp was run successfully."
        )

    citation_cols = [f"citations_{y}" for y in range(start_year, end_year + 1)]

    def _row_citations(row: pd.Series) -> pd.Series:
        cby_list = parse_counts_by_year_json(row.get("openalex_counts_by_year_json"))
        cit_dict = compute_cumulative_citations(
            counts_by_year=cby_list,
            start_year=start_year,
            end_year=end_year,
        )
        return pd.Series([cit_dict[c] for c in citation_cols], index=citation_cols)

    print("Computing cumulative citations per author from OpenAlex counts_by_year (resolved)...")
    citations_df = df_resolved[["author", "university", "openalex_counts_by_year_json"]].copy()
    citations_values = citations_df.apply(_row_citations, axis=1)

    result = pd.concat(
        [citations_df[["author", "university"]], citations_values],
        axis=1,
    )

    result[citation_cols] = result[citation_cols].fillna(0.0)
    return result


def build_citation_table_from_unresolved(unresolved_csv, start_year=2010, end_year=2020):
    """
    Read openalex_author_unresolved.csv and produce a small dataframe
    with (author, university) + cumulative citation columns for start_year to end_year.

    :param unresolved_csv: csv with manually-collected citations
    :param start_year: start year
    :param end_year: end year
    :return: small dataframe with final citation counts
    """

    df_unresolved = pd.read_csv(unresolved_csv)

    if "author" not in df_unresolved.columns or "university" not in df_unresolved.columns:
        raise ValueError("Expected 'author' and 'university' in unresolved CSV")

    if "citations_total" not in df_unresolved.columns:
        raise ValueError("Expected 'citations_total' in unresolved CSV")

    # We expect c_2010..c_2025; use all to compute baseline, but only output up to end_year.
    all_year_cols = [f"c_{y}" for y in range(start_year, 2026) if f"c_{y}" in df_unresolved.columns]

    df_unresolved["citations_total"] = pd.to_numeric(
        df_unresolved["citations_total"], errors="coerce"
    ).fillna(0.0)
    for col in all_year_cols:
        df_unresolved[col] = pd.to_numeric(df_unresolved[col], errors="coerce").fillna(0.0)

    citation_cols = [f"citations_{y}" for y in range(start_year, end_year + 1)]

    def _row_citations(row: pd.Series) -> pd.Series:
        # Sum of post-2010 yearly counts (2010..2025)
        sum_post = float(row[all_year_cols].sum())
        total = float(row["citations_total"])

        # Baseline = pre-2010 citations
        baseline = total - sum_post

        cumulative = baseline
        out = {}
        for year in range(start_year, end_year + 1):
            c_col = f"c_{year}"
            incr = float(row.get(c_col, 0.0)) if c_col in row.index else 0.0
            cumulative += incr
            out[f"citations_{year}"] = cumulative

        return pd.Series([out[c] for c in citation_cols], index=citation_cols)

    print("Computing cumulative citations per author from unresolved c_YYYY columns...")
    base_df = df_unresolved[["author", "university"] + all_year_cols + ["citations_total"]].copy()
    citations_values = base_df.apply(_row_citations, axis=1)

    result = pd.concat(
        [base_df[["author", "university"]], citations_values],
        axis=1,
    )

    result[citation_cols] = result[citation_cols].fillna(0.0)
    return result


def merge_citations_into_faculty(faculty_csv, resolved_csv, unresolved_csv, output_csv, start_year=2010, end_year=2020):
    """
    Stage 3:
      - Reads faculty_dec.csv
      - Reads openalex_author_candidates_resolved.csv (JSON counts_by_year)
      - Reads openalex_author_unresolved.csv (c_2010...c_2025 + citations_total)
      - Builds cumulative citation columns (citations_2010...citations_2020)
        for *both* resolved and unresolved authors.
      - Concatenates the two citation tables and merges them into faculty_df
        on (author, university).
      - Writes out a new faculty CSV with the citation columns appended.

    :param faculty_csv: faculty.csv
    :param resolved_csv: csv with OpenAlex citations
    :param unresolved_csv: csv with manually-collected citations
    :param output_csv: output csv
    :param start_year: start year
    :param end_year: end year
    :return: None
    """

    print(f"Loading faculty data from: {faculty_csv}")
    faculty_df = pd.read_csv(faculty_csv)

    if "author" not in faculty_df.columns or "university" not in faculty_df.columns:
        raise ValueError("Expected 'author' and 'university' columns in faculty_dec.csv")

    citation_cols = [f"citations_{y}" for y in range(start_year, end_year + 1)]
    for col in citation_cols:
        if col in faculty_df.columns:
            raise ValueError(
                f"Column '{col}' already exists in faculty_dec.csv. "
                "Refusing to overwrite; consider writing to a new file or renaming columns."
            )

    print(f"Loading resolved OpenAlex mapping from: {resolved_csv}")
    citations_resolved_df = build_citation_table_from_openalex(
        resolved_csv=resolved_csv,
        start_year=start_year,
        end_year=end_year,
    )

    print(f"Loading unresolved OpenAlex mapping from: {unresolved_csv}")
    citations_unresolved_df = build_citation_table_from_unresolved(
        unresolved_csv=unresolved_csv,
        start_year=start_year,
        end_year=end_year,
    )

    # Combine both; if there is overlap on (author, university),
    # unresolved entries will overwrite resolved (keep last).
    print("Combining resolved and unresolved citation tables...")
    combined_citations = pd.concat(
        [citations_resolved_df, citations_unresolved_df],
        ignore_index=True,
    )
    combined_citations = combined_citations.sort_values(
        by=["author", "university"]
    ).drop_duplicates(subset=["author", "university"], keep="last")

    print("Merging citation columns into faculty dataframe...")
    merged = faculty_df.merge(
        combined_citations,
        on=["author", "university"],
        how="left",
    )

    # Any rows with no OpenAlex info at all → NaN; treat as 0 citations.
    merged[citation_cols] = merged[citation_cols].fillna(0.0)

    print(
        merged[
            ["author", "citations_2010", "citations_2011", "citations_2012", "citations_2013"]
        ].head()
    )

    print(f"Saving updated faculty data with citation columns to: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    # Quick summary
    num_rows = len(merged)
    num_with_counts = (merged[citation_cols].sum(axis=1) > 0).sum()
    print(f"Total faculty rows: {num_rows}")
    print(f"Rows with any non-zero citations (2010–2020): {num_with_counts}")


def main():
    """
    Example pipeline.

      1. Run Stage 1 once to build candidates.
      2. Run Stage 2 once (can be slow; lots of API calls).
      3. After adding c_YYYY and citations_total to unresolved CSV,
         run Stage 3 to merge citations into faculty_dec.
    """
    faculty_csv = Path("../data/hiring/faculty_dec.csv")
    candidates_out = Path("../data/hiring/openalex_author_candidates.csv")
    resolved_out = Path("../data/hiring/openalex_author_candidates_resolved.csv")
    unresolved_out = Path("../data/hiring/openalex_author_unresolved.csv")
    dblp_csv = Path("../data/coauthorship/dblp_cleaned/full_by_author.csv")
    faculty_with_cites = Path("../data/hiring/faculty_dec_with_citations.csv")

    # Stage 1: candidates
    build_author_candidate_table(
        faculty_csv=faculty_csv,
        output_csv=candidates_out,
    )

    # Stage 2: resolve + c_by_y
    resolve_candidates_with_dblp(
        candidates_csv=candidates_out,
        dblp_csv=dblp_csv,
        output_csv=resolved_out,
        max_works_per_candidate=100,
    )
    export_unresolved_authors(
        resolved_csv=resolved_out,
        unresolved_out=unresolved_out,
    )

    # Stage 3: cumulative citations + merge
    # NOTE: assumes unresolved_out has been augmented with c_YYYY + citations_total.
    merge_citations_into_faculty(
        faculty_csv=faculty_csv,
        resolved_csv=resolved_out,
        unresolved_csv=unresolved_out,
        output_csv=faculty_with_cites,
        start_year=2010,
        end_year=2020,
    )


if __name__ == "__main__":
    main()
