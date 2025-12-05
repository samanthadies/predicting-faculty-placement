"""
format_dblp.py

This script cleans and reshapes DBLP-derived CSVs into:
1) paper-level tables (optionally filtered to papers with at least one faculty)
2) author-level tables exploded by author with positional order
3) a faculty table with university rankings, doctoral rankings, and derived labels y

Expected input CSVs:
- Raw per-element DBLP CSVs (semicolon-delimited) at: <base_fp><pub_type>.csv
  where pub_type is one of {'proceedings', 'inproceedings', 'article'}.
- Faculty roster at '../../data/hiring/faculty_raw.csv' (https://jeffhuang.com/computer-science-open-data/)
- University rankings at '../../data/university_ranking/csrankings.csv' (https://csrankings.org/)

Outputs:
- '../../data/coauthorship/dblp_clean/full_by_paper.csv'
- '../../data/coauthorship/dblp_clean/full_w_faculty_by_paper.csv'
- '../../data/coauthorship/dblp_cleaned/full_by_author.csv'
- '../../data/hiring/faculty.csv'

Notes:
- Authors are stored as pipe-delimited strings in input and converted to lists as needed.
- Multi-author rows are exploded to one row per (paper, author) with 1-based `author_order`.
- Ranking-based label y: 0 (top 1–10), 1 (11–50), 2 (>50).
- Uses `tqdm.pandas()` to show progress when scanning authors.

10/24/2025 — SD
"""

import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()   # enable progress bars for pandas .progress_apply


def get_faculty_only(df, faculty_list):
    """
    Filter paper rows to those that include at least one faculty member.

    :param df: Paper-level dataframe containing an 'author' column as a pipe-delimited string (pd.DataFrame)
    :param faculty_list: List of faculty names to match (substring match per author field) (Sequence[str])
    :return: Subset of rows where at least one author matches `faculty_list` (pd.DataFrame)
    """

    def check_author(authors):
        author_names = authors.split('|')  # Split the author names

        # identify publications which have at least one of our faculty members
        for name in faculty_list:
            if any(name in author for author in author_names):
                return True
        return False

    # check for faculty members and drop papers without any of our faculty
    faculty_only_df = df[df['author'].progress_apply(check_author)]
    faculty_only_df = faculty_only_df.reset_index().drop(['index'], axis=1)

    return faculty_only_df


def preprocess(df, pub_type, faculty_list):
    """
    Standardize raw per-type DBLP tables and produce:
      (a) all papers for this type
      (b) only papers with at least one faculty author

    :param df: Raw per-type DBLP dataframe (pd.DataFrame)
    :param pub_type: One of {'proceedings','inproceedings','article'} (str)
    :param faculty_list: List of faculty names for filtering (Sequence[str])
    :return: (all_papers_df, faculty_only_papers_df) with standardized columns:
                ['year','conference_journal','num_authors','author','publisher','series','pub_type']
             (pd.DataFrame, pd.DataFrame)
    """
    # identify subset of columns depending on the publication type
    if pub_type == 'proceedings':
        # note: authors listed in 'editor', conference listed as 'booktitle'
        df = df[['id', 'editor', 'year', 'booktitle', 'publisher', 'series']]
        df = df.rename(columns={'editor': 'author', 'booktitle': 'conference_journal'})
    elif pub_type == 'inproceedings':
        # note: conference listed as 'booktitle'
        df = df[['id', 'author', 'booktitle', 'year']]
        df = df.rename(columns={'booktitle': 'conference_journal'})
        df[['publisher', 'series']] = np.nan
    else:
        df = df[['id', 'author', 'journal', 'publisher', 'year']]
        df = df.rename(columns={'journal': 'conference_journal'})
        df['series'] = np.nan

    # drop rows with null values for author and year and rename columns
    df = df.dropna(subset=['author', 'year'])
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    df['year'] = df['year'].astype(int)

    # reformat author column to hold lists of authors instead of strings
    df['num_authors'] = df['author'].str.count('\|') + 1
    df['num_authors'] = df['num_authors'] + 1

    # make sure we only keep papers from faculty in our faculty df
    faculty_only_pubs_df = get_faculty_only(df, faculty_list)

    # reformat author names
    df['author'] = df.author.str.split('\|', expand=False)

    # append the pub_type as a column
    pubs = pd.Series([pub_type]).repeat(len(df)).reset_index().drop(['index'], axis=1)
    df['pub_type'] = pubs[0]
    faculty_pub_type = [pub_type] * (len(faculty_only_pubs_df.index))
    faculty_only_pubs_df['pub_type'] = faculty_pub_type

    # reorder columns and save paper-level dataframe
    df = df[['year', 'conference_journal', 'num_authors', 'author', 'publisher', 'series', 'pub_type']]
    faculty_only_pubs_df = faculty_only_pubs_df[['year', 'conference_journal', 'num_authors', 'author', 'publisher',
                                                 'series', 'pub_type']]

    return df, faculty_only_pubs_df


def format_paper_level(base_fp, faculty_list):
    """
    Build combined paper-level tables across publication types.

    Writes:
      - '../../data/coauthorship/dblp_clean/full_by_paper.csv'
      - '../../data/coauthorship/dblp_clean/full_w_faculty_by_paper.csv'

    :param base_fp: Directory/prefix where raw per-type CSVs live; expects files: f'{base_fp}proceedings.csv',
                    f'{base_fp}inproceedings.csv', f'{base_fp}article.csv' (str)
    :param faculty_list: List of faculty names for filtering (Sequence[str])
    :return: None
    """
    pub_types = ['proceedings', 'inproceedings', 'article']
    full_df = pd.DataFrame(columns=['year', 'conference_journal', 'num_authors', 'author', 'publisher', 'series',
                                    'pub_type'])
    with_faculty_df = pd.DataFrame(columns=['year', 'conference_journal', 'num_authors', 'author', 'publisher',
                                            'series', 'pub_type'])

    for list in pub_types:
        print('\n\n\nPUB TYPE: ' + str(list) + '\n\n\n')
        # preprocess/clean file and save paper-level data
        fp = base_fp + list + '.csv'
        raw_df = pd.read_csv(fp, delimiter=';')
        df, faculty_df = preprocess(raw_df, list, faculty_list)

        # add to full list of publications
        full_df = pd.concat([full_df, df])
        with_faculty_df = pd.concat([with_faculty_df, faculty_df])

    # add relevant features
    full_df['paper_id'] = full_df.index + 1
    full_df = full_df[['paper_id', 'pub_type', 'year', 'conference_journal', 'num_authors', 'author', 'publisher',
                       'series']]
    full_df[['num_authors', 'year']] = full_df[['num_authors', 'year']].astype(int)

    with_faculty_df['paper_id'] = with_faculty_df.index + 1
    with_faculty_df = with_faculty_df[['paper_id', 'pub_type', 'year', 'conference_journal', 'num_authors', 'author',
                                       'publisher', 'series']]
    with_faculty_df[['num_authors', 'year']] = with_faculty_df[['num_authors', 'year']].astype(int)

    # save data
    full_df.to_csv('../../data/coauthorship/dblp_clean/full_by_paper.csv', index=False)
    with_faculty_df.to_csv('../../data/coauthorship/dblp_clean/full_w_faculty_by_paper.csv', index=False)


def format_author_level(df, faculty):
    """
    Build author-level and faculty-level tables.

    Steps
    1) Explode by author with 1-based order per paper.
    2) Merge with faculty roster to attach metadata (join year, institutions, etc.).
    3) Drop authors without a hire/join year.
    4) Create a de-duplicated faculty table and augment with rankings and indicators.
    5) Persist both author-level and faculty-level tables.

    :param df: Paper-level dataframe containing at least:
               ['paper_id','year','author','num_authors','pub_type','title','journal','id','author_list'] (as used below)
               (pd.DataFrame)
    :param faculty: Faculty roster with columns including:
                    ['FullName','University','JoinYear','SubField','Bachelors','Doctorate']
                    (pd.DataFrame)
    :return: None
    """
    # explode df into one by authors including the author order
    author_df = explode_authors_with_order(df)

    # merge faculty and author data
    df_merged = author_df.merge(
        faculty,
        left_on='author',
        right_on='FullName',
        how='left'
    )

    # drop authors if we don't have a hire year
    df_merged = df_merged.dropna(subset=['JoinYear']).reset_index().drop(columns=['index'])
    df_merged['JoinYear'] = df_merged['JoinYear'].astype(int)

    # rename columns
    df_merged = df_merged.rename(columns={'University': 'university', 'JoinYear': 'join_year', 'SubField': 'sub_field',
                                          'Bachelors': 'bachelors', 'Doctorate': 'doctorate'})
    df_merged = df_merged.drop(columns=['FullName'])
    df_merged = df_merged[['author', 'join_year', 'doctorate', 'university', 'bachelors', 'sub_field', 'year',
                           'author_order', 'paper_id', 'id', 'num_authors', 'pub_type', 'title', 'journal',
                           'author_list']]

    df_merged = df_merged[df_merged['year'] < 2023]
    df_merged = df_merged[df_merged['join_year'] < 2023]

    # format faculty df by dropping duplicate authors
    df_faculty = df_merged.drop_duplicates(subset=['author'])
    df_faculty = df_faculty.drop(columns=['year', 'author_order', 'paper_id', 'id', 'num_authors', 'pub_type', 'title',
                                          'journal', 'author_list'])

    # add in university rankings and generate y
    df_faculty = get_rankings(df_faculty)

    # add info about faculty
    df_faculty = get_faculty_indicators(df_faculty)
    df_faculty = get_academic_age(df_faculty)

    print(df_faculty.info())

    # save faculty and author-level dataframes
    df_merged.to_csv('../../data/coauthorship/dblp_cleaned/full_by_author.csv', index=False)
    df_faculty.to_csv('../../data/hiring/faculty.csv', index=False)


def get_rankings(fac_df):
    """
    Merge CSRankings for (current) university and doctoral institution, compute label y.

    :param fac_df: Faculty dataframe with 'university' and 'doctorate' columns (pd.DataFrame)
    :return: Faculty dataframe augmented with 'ranking', 'doctoral_ranking', and categorical 'y' (pd.DataFrame)
    """
    # add in the ranking of the faculty university
    ranking_df = pd.read_csv('../../data/university_ranking/csrankings.csv')
    ranking_df = ranking_df.drop_duplicates(subset='University')
    fac_df = fac_df.merge(ranking_df[['University', 'Ranking']], left_on='university', right_on='University',
                          how='left')
    fac_df = fac_df.rename(columns={'Ranking': 'ranking'})
    fac_df = fac_df.drop(columns=['University'])
    avg_rank = fac_df['ranking'].mean()
    fac_df['ranking'] = fac_df['ranking'].fillna(avg_rank)
    fac_df['ranking'] = fac_df['ranking'].round().astype(int)

    # add in the ranking of the doctoral university
    fac_df = fac_df.merge(ranking_df[['University', 'Ranking']], left_on='doctorate', right_on='University', how='left')
    fac_df = fac_df.rename(columns={'Ranking': 'doctoral_ranking'})
    fac_df = fac_df.drop(columns=['University'])
    avg_rank = fac_df['doctoral_ranking'].mean()
    fac_df['doctoral_ranking'] = fac_df['doctoral_ranking'].fillna(avg_rank)
    fac_df['doctoral_ranking'] = fac_df['doctoral_ranking'].round().astype(int)

    # set up y
    conditions = [
        (fac_df['ranking'] >= 1) & (fac_df['ranking'] <= 10),
        (fac_df['ranking'] >= 11) & (fac_df['ranking'] <= 50),
        (fac_df['ranking'] > 50)
    ]
    choices = [0, 1, 2]
    fac_df['y'] = np.select(conditions, choices, default=2)
    fac_df['y'] = fac_df['y'].astype(int)

    return fac_df


def get_faculty_indicators(fac_df):
    """
    Add year-wise faculty indicators: fac_{year} = 1 if join_year <= year else 0.

    :param fac_df: Faculty dataframe with 'join_year' column (pd.DataFrame)
    :return: Faculty dataframe with added indicator columns fac_2010...fac_2020 (pd.DataFrame)
    """
    # add indicator-columns which capture whether they're faculty in a given year
    for year in range(2010, 2021):
        col_name = f'fac_{year}'
        fac_df[col_name] = (fac_df['join_year'] <= year).astype(int)

    return fac_df


def get_academic_age(fac_df):
    """
    Add year-wise academic age indicators: academic_age_{year} = year - phd_year if phd_year <= year, else -1.

    :param fac_df: Faculty dataframe (pd.DataFrame)
    :return: Faculty dataframe with added academic_age columns academic_age_2010...academic_age_2020 (pd.DataFrame)
    """
    # Make sure phd_year and join_year are numeric
    fac_df["phd_year"] = pd.to_numeric(fac_df["phd_year"], errors="coerce")
    fac_df["join_year"] = pd.to_numeric(fac_df["join_year"], errors="coerce")

    for year in range(2010, 2021):
        fac_col = f"fac_{year}"
        age_col = f"academic_age_{year}"

        phd = fac_df["phd_year"]
        join = fac_df["join_year"]
        fac = fac_df[fac_col]

        # Start with -1 everywhere (pre-PhD / unknown)
        age = pd.Series(-1, index=fac_df.index, dtype="Int64")

        # Have a known PhD year, and it's already been granted: use year - phd_year
        mask_has_phd_and_reached = phd.notna() & (phd <= year)
        age[mask_has_phd_and_reached] = (year - phd)[mask_has_phd_and_reached]

        # No PhD year, but already faculty in this year: use join_year as proxy
        mask_no_phd_but_faculty = phd.isna() & (fac == 1) & join.notna()
        age[mask_no_phd_but_faculty] = (year - join)[mask_no_phd_but_faculty]

        # Everyone else stays at -1
        fac_df[age_col] = age

    return fac_df


def explode_authors_with_order(df):
    """
    Explode pipe-delimited 'author' strings to one row per author with a 1-based order.

    :param df: Paper-level dataframe containing columns
               ['paper_id','id','author','num_authors','pub_type','title','journal','year']
               (pd.DataFrame)
    :return: Author-level rows with columns:
             ['author','year','author_order','paper_id','id','num_authors','pub_type','title','journal','author_list']
             (pd.DataFrame)
    """
    df = df.rename(columns={'author': 'author_list'})
    df['author'] = df['author_list'].str.split('|')

    df_exploded = df.explode('author').reset_index(drop=True)
    # 1 author order per paper
    df_exploded['author_order'] = df_exploded.groupby('paper_id').cumcount() + 1

    # Keep only the columns that exist in df_exploded (some datasets may not have title/journal)
    df_exploded = df_exploded[['author', 'year', 'author_order', 'paper_id', 'id', 'num_authors', 'pub_type', 'title',
                               'journal', 'author_list']]

    return df_exploded


def main():
    """
    Build author/faculty tables from pre-computed paper-level CSVs.

    - To generate paper-level tables from raw per-type CSVs:
        1) Load faculty list
        2) Call format_paper_level(base_fp, faculty_list)

    - To build author-level & faculty tables from the cleaned paper-level CSV:
        1) Read '../../data/coauthorship/dblp_cleaned/full_by_paper.csv'
        2) Call format_author_level(df, faculty)

    :return: None
    """
    warnings.filterwarnings("ignore")

    # Faculty roster (expected columns: FullName, University, JoinYear, SubField, Bachelors, Doctorate, ...)
    faculty = pd.read_csv('../../data/hiring/faculty_raw.csv')

    # Generate the paper-level cleaned tables from raw per-type CSVs
    faculty_list = faculty['author'].to_list()
    base_fp = '../../data/coauthorship/dblp_raw/'
    format_paper_level(base_fp, faculty_list)

    # Build author-level df and faculty df from the cleaned paper-level CSV
    df = pd.read_csv('../../data/coauthorship/dblp_cleaned/full_by_paper.csv')
    format_author_level(df, faculty)


if __name__ == '__main__':
    main()
