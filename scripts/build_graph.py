"""
build_graph.py

This script generates consistent train/val/test splits once and save them to disk. It
also builds author–paper incidence matrices and co-authorship adjacency matrices.

Inputs:
- '../data/coauthorship/dblp_cleaned/full_by_author.csv'
    One row per (paper, author) with columns including:
    ['paper_id', 'author', 'year', ...]
- '../../data/hiring/faculty.csv'
    Faculty roster with a column 'author' (canonicalized full name).

Outputs:
Splits:
- '../data/hiring/faculty_{train,val,test}.csv' (built by `get_train_test()`)

Bipartite (author x paper) matrices:
- '../data/graph_bipartite/non_cumulative_bipartite_{YEAR}.npz'  (dense, saved via np.savez)
    Per-year incidence (2010 uses <= 2010; later years use == YEAR)
- '../data/graph_bipartite/bipartite_{YEAR}.npz'                 (dense, saved via np.savez)
    Cumulative incidence up to YEAR (sum of prior years' binary matrices)

Coauthorship (author x author) matrices (sparse CSR):
- '../data/graph_adjmats/non_cumulative_adjmat_{YEAR}.npz'       (scipy.sparse.save_npz)
    Per-year coauthorship via B @ B.T
- '../data/graph_adjmats/adjmat_{YEAR}.npz'                      (scipy.sparse.save_npz)
    Cumulative coauthorship (sum over years <= YEAR)

Notes:
- Author universe (rows) is fixed to faculty_df['author'] order.
- Paper universe (cols) is fixed to all unique paper_ids in `paper_df`.
- Incidence is binary (1 if (author, paper) observed in the selected slice).
- Non-cumulative year 2010 includes all papers with year ≤ 2010 (bootstrap year);
  later non-cumulative years include only papers with year == YEAR.

1/13/2026 — SD
"""

import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split

YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]


def get_train_test():
    """
    Create and persist train/val/test splits from faculty.csv.

    Rules:
      - Keep hires between 2010 and 2020 (inclusive) as ML cohort.
      - Test = join_year > 2017 (i.e., 2018...2020)
      - Train/Val = join_year <= 2017. For each hire year in 2010...2017, 80/20 split.

    Writes:
      ../data/hiring/faculty_train.csv
      ../data/hiring/faculty_val.csv
      ../data/hiring/faculty_test.csv

    :return: None
    """
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')
    faculty_df = faculty_df.reset_index().rename(columns={'index': 'fac_idx'})

    # only consider those hired between 2010 and 2020
    ml_df = faculty_df[(faculty_df['join_year'] >= 2010) & (faculty_df['join_year'] <= 2020)].copy()

    # get the test df
    test_df = ml_df[ml_df['join_year'] > 2017]

    # get the train/val df
    train_val_df = ml_df[ml_df['join_year'] <= 2017]

    # do an 80/20 split between train and val for each hire year
    val_indices = []
    for year in range(2010, 2018):
        year_group = train_val_df[train_val_df['join_year'] == year]
        _, val_subset = train_test_split(year_group, test_size=0.2, random_state=42, shuffle=True)
        val_indices.extend(val_subset.index.tolist())

    # get final train/val splits
    val_df = train_val_df.loc[val_indices]
    train_df = train_val_df.drop(val_indices)

    # save
    train_df.to_csv('../data/hiring/faculty_train.csv', index=False)
    val_df.to_csv('../data/hiring/faculty_val.csv', index=False)
    test_df.to_csv('../data/hiring/faculty_test.csv', index=False)


def author_by_paper(paper_df, faculty_df, year, all_paper_ids):
    """
    Construct a binary author×paper incidence matrix for a given year.

    For 2010: include all rows with year <= 2010.
    For later years: include only rows with year == {year}.

    :param paper_df: Author-level table with at least ['author', 'paper_id', 'year'].
                     Typically loaded from '../data/coauthorship/dblp_cleaned/full_by_author.csv'.
                     (pd.DataFrame)
    :param faculty_df: Faculty table with a column 'author' defining the row index (pd.DataFrame)
    :param year: Target year for the non-cumulative slice (int)
    :param all_paper_ids: Full set of paper IDs to serve as columns (consistent across years) (Sequence[int])
    :return: DataFrame of shape [n_authors, n_papers] with {0,1} entries.
             Index = faculty authors; columns = all_paper_ids.
             (pd.DataFrame)
    """
    # Select the per-year slice (special handling for 2010)
    if year == 2010:
        df = paper_df[paper_df['year'] <= year]
    else:
        df = paper_df[paper_df['year'] == year]

    # Fixed author and paper universes
    author_list = faculty_df['author'].tolist()
    bipartite = pd.DataFrame(0, index=author_list, columns=all_paper_ids)

    # Construct incidence matrix (author x paper)
    for _, row in df.iterrows():
        author = row['author']
        paper = row['paper_id']
        if author in bipartite.index and paper in bipartite.columns:
            bipartite.at[author, paper] = 1

    return bipartite


def build_bipartite_adjmats(paper_df, faculty_df):
    """
    Build and persist per-year (non-cumulative) and cumulative author×paper matrices.

    Non-cumulative:
      '../data/graph_bipartite/non_cumulative_bipartite_{YEAR}.npz'
    Cumulative:
      '../data/graph_bipartite/bipartite_{YEAR}.npz'

    :param paper_df: Author-level table with ['author','paper_id','year'] (pd.DataFrame)
    :param faculty_df: Faculty table with ['author'] (defining row order) (pd.DataFrame)
    :return: None
    """
    all_paper_ids = sorted(paper_df['paper_id'].unique().tolist())

    # Per-year (non-cumulative) incidence matrices (kept in memory to build cumulative later)
    year_bipartites = []
    for year in YEARS:
        bipartite = author_by_paper(paper_df, faculty_df, year, all_paper_ids)
        bipartite = bipartite.to_numpy()
        year_bipartites.append(bipartite)
        # Save dense array (np.savez, not compressed)
        np.savez(f'../data/graph_bipartite/non_cumulative_bipartite_{year}.npz', adjmat=bipartite)

    # Cumulative author x paper incidence: sum of prior year slices
    prev = None
    for i, bip in enumerate(year_bipartites):
        cumulative = bip if prev is None else (prev + bip)
        np.savez(f"../data/graph_bipartite/bipartite_{YEARS[i]}.npz", adjmat=cumulative)
        prev = cumulative


def build_coauthorship_adjmats():
    """
    Build and persist per-year (non-cumulative) and cumulative coauthorship adjacency.

    Non-cumulative (per year):
      - Load '../data/graph_bipartite/non_cumulative_bipartite_{YEAR}.npz'
      - Compute A_year = B_year @ B_year.T  (sparse CSR)
      - Save to '../data/graph_adjmats/non_cumulative_adjmat_{YEAR}.npz'

    Cumulative:
      - cumulative_A_YEAR = sum_{t ≤ YEAR} A_t
      - Save to '../data/graph_adjmats/adjmat_{YEAR}.npz'

    :return: None
    """
    year_adjmats = []

    for year in tqdm(YEARS):
        print(year)

        # Load dense bipartite for this year
        file = np.load(f'../data/graph_bipartite/non_cumulative_bipartite_{year}.npz', allow_pickle=True)
        bipartite = file['adjmat']

        # Convert to sparse and compute coauthorship A = B B^T
        B = csr_matrix(bipartite)
        A = B @ B.T  # author x author
        year_adjmats.append(A)
        print("\t", A.shape)

        save_npz(f'../data/graph_adjmats/non_cumulative_adjmat_{year}.npz', A)

    # Cumulative coauthorship = sum of annual adjacencies
    prev = None
    for i, A in enumerate(year_adjmats):
        cumulative = A if prev is None else (prev + A)
        save_npz(f"../data/graph_adjmats/adjmat_{YEARS[i]}.npz", cumulative)
        prev = cumulative


def main():
    """
    Orchestrate the full pipeline:

        1) Generate consistent train/test splits for ML.
        1) Read author-level paper data and faculty roster.
        2) Build & save non-cumulative and cumulative author x paper matrices.
        3) Build & save non-cumulative and cumulative coauthorship adjacency.

    :return: None
    """
    warnings.filterwarnings('ignore')

    # Generate train/test splits
    get_train_test()

    paper_df = pd.read_csv('../data/coauthorship/dblp_cleaned/full_by_author.csv')
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')

    # Build and save author x paper incidence matrices
    build_bipartite_adjmats(paper_df, faculty_df)

    # Build and save co-authorship adjacency matrices
    build_coauthorship_adjmats()


if __name__ == '__main__':
    main()