"""
get_bibliometric_features.py

Bibliometric feature construction from bipartite and author-level DBLP tables.
1) For each year (2010..2020), load the (author x paper) non-cumulative
   bipartite incidence matrix and compute:
   - num_papers (per author)
   - avg_authors_per_paper (per author)
   - first-author features: num_first_authored, avg_author_position, prop_first_authored
   - prestige features of coauthors (faculty presence and ranking tiers)

2) Aggregate to author-level feature snapshots per year, then transform to:
   - cumulative features up to (but not including) each author's join year
     (sums for counts; weighted means for proportions/averages)
   - previous-year features (or -1 if undefined)

3) Merge into faculty.csv in-place.

1/13/2026 — SD
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def load_bipartite_matrix(year):
    """
    Load the (author x paper) non-cumulative bipartite matrix for a given year.

    For 2010, it contains authors' papers with year <= 2010.
    For later years, it contains authors' papers with year == year.

    :param year: the year (int)
    :return: array of shape (n_authors, n_papers) (np.ndarray)
    """
    file = np.load(f'../data/graph_bipartite/non_cumulative_bipartite_{year}.npz', allow_pickle=True)
    bipartite = file['adjmat']

    return bipartite


def first_author_features(year, df):
    """
    Compute first-author–related metrics for a given year.

    :param year: the year (int)
    :param df: the author-level dataframe
    :return: num_first_authored: per-author count of papers where author_order == 1
             avg_author_position: per-author mean author_order
             prop_first_authored: per-author share of first-authored papers
    """
    if year == 2010:
        df_year = df[df['year'] <= year]
    else:
        df_year = df[df['year'] == year]

    # count the number of first-authored papers
    num_first_authored = (
        df_year[df_year['author_order'] == 1]
        .groupby('author')['paper_id']
        .nunique()
        .rename('num_first_authored')
    )

    # find the average author position
    avg_author_position = (
        df_year.groupby('author')['author_order']
        .mean()
        .rename('avg_author_position')
    )

    # find the proportion of first-authored papers (# first-authored) / (# total papers)
    total_papers = df_year.groupby('author')['paper_id'].nunique()
    prop_first_authored = (
        (num_first_authored / total_papers)
        .fillna(0)
        .rename('prop_first_authored')
    )

    return num_first_authored, avg_author_position, prop_first_authored


def author_prestige_features(year, df, faculty_df):
    """
    Coauthor prestige features per author for the given year.

    For each (author, paper), inspect co-authors on that paper and mark whether
    any coauthor is:
      - faculty (fac_<year> == 1)
      - top faculty (fac_<year> == 1 and y == 0)
      - mid faculty (fac_<year> == 1 and y == 1)

    Produce counts and proportions over the author's papers.

    :param year: the year (int)
    :param df: the author-level dataframe
    :param faculty_df: the faculty dataframe
    :return: dictionary of co-author features
    """
    if year == 2010:
        df_year = df[df['year'] <= year]
    else:
        df_year = df[df['year'] == year]

    # figure out whether the authors are faculty in that year
    fac_col = f'fac_{year}'
    faculty_lookup = faculty_df.set_index('author')[[fac_col, 'y']]
    df_year = df_year.merge(faculty_lookup, how='left', left_on='author', right_index=True)

    # grab all the info about the authors of the papers (except our focal author)
    paper_to_coauthors = defaultdict(list)
    for pid, group in df_year.groupby('paper_id'):
        author_rows = group.to_dict('records')
        for row in author_rows:
            coauthors = [r for r in author_rows if r['author'] != row['author']]
            paper_to_coauthors[(row['author'], pid)] = coauthors

    author_paper_groups = df_year.groupby('author')['paper_id'].unique()

    coauth_features = {
        'num_faculty_papers': {},
        'prop_faculty_papers': {},
        'num_high_faculty_papers': {},
        'prop_high_faculty_papers': {},
        'num_med_faculty_papers': {},
        'prop_med_faculty_papers': {},
    }

    # calculate features
    for author, paper_ids in author_paper_groups.items():
        n_pubs = len(paper_ids)
        n_fac, n_top, n_mid = 0, 0, 0

        for pid in paper_ids:
            coauthors = paper_to_coauthors.get((author, pid), [])

            has_faculty = any(c.get(fac_col) == 1 for c in coauthors)
            has_top = any(c.get(fac_col) == 1 and c.get('y') == 0 for c in coauthors)
            has_mid = any(c.get(fac_col) == 1 and c.get('y') == 1 for c in coauthors)

            n_fac += int(has_faculty)
            n_top += int(has_top)
            n_mid += int(has_mid)

        # fill the feature dict
        coauth_features['num_faculty_papers'][author] = n_fac
        coauth_features['prop_faculty_papers'][author] = n_fac / n_pubs if n_pubs > 0 else 0

        coauth_features['num_high_faculty_papers'][author] = n_top
        coauth_features['prop_high_faculty_papers'][author] = n_top / n_pubs if n_pubs > 0 else 0

        coauth_features['num_med_faculty_papers'][author] = n_mid
        coauth_features['prop_med_faculty_papers'][author] = n_mid / n_pubs if n_pubs > 0 else 0

    return coauth_features


def get_bibliometric_features():
    """
    Build per-year bibliometric features for all authors.

    :return: dictionary of bibliometric features
    """
    full_by_author = pd.read_csv('../data/coauthorship/dblp_cleaned/full_by_author.csv')
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')
    author_ids = faculty_df['author'].tolist()
    all_features = {}

    for year in tqdm(range(2010, 2021), desc='Building feature sets'):
        print(f'year: {year}')
        bipartite = load_bipartite_matrix(year)

        # number of papers
        num_papers = pd.Series(bipartite.sum(axis=1), index=author_ids, name='num_papers')
        print('\tgot num papers')

        # average authors per paper
        paper_author_counts = bipartite.sum(axis=0)
        avg_authors_raw = (bipartite @ paper_author_counts)
        denom = bipartite.sum(axis=1).astype(float)
        avg_authors_vals = np.divide(
            avg_authors_raw,
            denom,
            out=np.zeros_like(avg_authors_raw, dtype=float),  # fill 0 where denom == 0
            where=denom != 0
        )
        avg_authors = pd.Series(avg_authors_vals, index=author_ids, name='avg_authors_per_paper')
        print('\tgot avg authors')

        # features relating to author order
        num_first_authored, avg_author_position, prop_first_authored = first_author_features(year, full_by_author)
        print('\tgot author order features')

        # features relating to prestige
        prestige_features = author_prestige_features(year, full_by_author, faculty_df)
        print('\tgot prestige features')

        # add to dictionary
        all_features[year] = {
            'num_papers': num_papers,
            'avg_authors_per_paper': avg_authors,
            'num_first_authored': num_first_authored,
            'avg_author_position': avg_author_position,
            'prop_first_authored': prop_first_authored,
        }

        for feat_name, feat_dict in prestige_features.items():
            all_features[year][feat_name] = pd.Series(feat_dict)

    return all_features


def add_citation_cum_pre_features(faculty_df):
    """
    Add cum_num_citations and pre_num_citations directly from citations_<year> columns.

    :param faculty_df: faculty-level dataframe
    :return: dataframe with new citation columns
    """

    df = faculty_df.copy()

    df["cum_num_citations"] = 0.0
    df["pre_num_citations"] = -1.0

    def get_cit(row, year):
        col = f"citations_{year}"
        val = row.get(col, 0.0)
        return float(val) if pd.notna(val) else 0.0

    for idx, row in df.iterrows():
        jy = row.get("join_year")
        if pd.isna(jy):
            continue

        try:
            jy = int(jy)
        except Exception:
            continue

        if 2012 <= jy <= 2020:
            c_prev = get_cit(row, jy - 1)
            c_prev2 = get_cit(row, jy - 2)
            df.at[idx, "cum_num_citations"] = c_prev
            df.at[idx, "pre_num_citations"] = max(0.0, c_prev - c_prev2)

        elif jy == 2011:
            df.at[idx, "cum_num_citations"] = get_cit(row, 2010)
            df.at[idx, "pre_num_citations"] = -1.0

        else:
            df.at[idx, "cum_num_citations"] = get_cit(row, 2020)
            df.at[idx, "pre_num_citations"] = -1.0

    return df


def add_academic_age_pre_features(faculty_df):
    """
    Add pre_academic_age directly from academic_age_<year> columns.

    :param faculty_df: faculty-level dataframe
    :return: dataframe with new academic age column
    """

    df = faculty_df.copy()

    df["pre_academic_age"] = -1.0

    def get_ac_age(row, year):
        col = f"academic_age_{year}"
        val = row.get(col, -1.0)
        return float(val) if pd.notna(val) else -1.0

    for idx, row in df.iterrows():
        jy = row.get("join_year")
        if pd.isna(jy):
            continue

        try:
            jy = int(jy)
        except Exception:
            continue

        if 2011 <= jy <= 2020:
            a_prev = get_ac_age(row, jy - 1)
            df.at[idx, "pre_academic_age"] = a_prev

        else:
            df.at[idx, "pre_academic_age"] = -1.0

    return df


def convert_to_usable(all_features):
    """
    Convert per-year features to:
      - cumulative up to join_year (exclusive)
      - previous-year snapshot
    and merge into faculty.csv.

    :param all_features: dictionary of year-level features
    :return: None
    """
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')

    feature_names = [
        'num_papers', 'avg_authors_per_paper', 'num_first_authored', 'avg_author_position',
        'prop_first_authored', 'num_faculty_papers', 'prop_faculty_papers',
        'num_high_faculty_papers', 'prop_high_faculty_papers',
        'num_med_faculty_papers', 'prop_med_faculty_papers'
    ]

    cumulative_feature_data = defaultdict(dict)
    prevyear_feature_data = defaultdict(dict)

    for _, row in tqdm(faculty_df.iterrows(), total=len(faculty_df), desc='Processing faculty'):
        print(f'{_}')
        author = row['author']
        join_year = row['join_year']
        print(f'\tjoin year: {join_year}')

        # Define relevant years
        if join_year <= 2010 or join_year > 2020:
            cumulative_years = list(range(2010, 2021))
            prevyear = None  # Use -1 values
            print(f'\tcumulative_years: {cumulative_years}')
            print(f'\tprevyear is None')
        else:
            cumulative_years = list(range(2010, join_year))
            prevyear = join_year - 1 if join_year > 2010 else None
            print(f'\tcumulative_years: {cumulative_years}')
            print(f'\tprevyear: {prevyear}')

        # Aggregate cumulative features (sum or mean depending on feature)
        for feat in feature_names:
            values = []
            for year in cumulative_years:
                val = all_features.get(year, {}).get(feat, {}).get(author, np.nan)
                values.append(val)

            if 'prop' in feat or 'avg' in feat:
                weighted_vals = []
                weights = []

                for year in cumulative_years:
                    val = all_features.get(year, {}).get(feat, {}).get(author, np.nan)
                    weight = all_features.get(year, {}).get('num_papers', {}).get(author, np.nan)
                    if not np.isnan(val) and not np.isnan(weight):
                        weighted_vals.append(val * weight)
                        weights.append(weight)

                if weights:
                    agg_val = np.nansum(weighted_vals) / np.nansum(weights)
                else:
                    agg_val = np.nan
            else:
                agg_val = np.nansum(values) if values else 0

            cumulative_feature_data[author][f'cum_{feat}'] = agg_val

        # Grab previous year value
        for feat in feature_names:
            if prevyear is None:
                prev_val = -1
            else:
                prev_val = all_features.get(prevyear, {}).get(feat, {}).get(author, -1)

            prevyear_feature_data[author][f'pre_{feat}'] = prev_val

    # Convert to DataFrames
    cum_df = pd.DataFrame.from_dict(cumulative_feature_data, orient='index')
    pre_df = pd.DataFrame.from_dict(prevyear_feature_data, orient='index')

    # Merge into faculty_df
    faculty_df = faculty_df.set_index('author')
    faculty_df = faculty_df.join(cum_df).join(pre_df)
    faculty_df = faculty_df.reset_index()

    # Add citation-derived features without altering the per-year all_features pipeline
    faculty_df = add_citation_cum_pre_features(faculty_df)

    # Add academic age features
    faculty_df = add_academic_age_pre_features(faculty_df)

    faculty_df = faculty_df.fillna(0)

    faculty_df.to_csv('../data/hiring/faculty.csv', index=False)


def main():
    """
    Generates bibliometric features.

    :return: None
    """

    all_features = get_bibliometric_features()
    convert_to_usable(all_features)


if __name__=='__main__':
    main()