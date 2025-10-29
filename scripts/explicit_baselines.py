"""
explicit_baselines.py

This script generates the random guessing, by avg. neighbor rank, and by PhD rank
heuristics for a given high-rank threshold.

10/29/2025 - SD
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def get_random_guessing(top=10):
    """
    Calculate random guessing baseline.

    :param top: the threshold for 'high' (int)
    :return: the random guessing baseline
    """
    test_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_test.csv')
    if top == 10:
        y = np.array(test_df['y'])
    else:
        y = np.array(test_df[f'y_{top}'])
    classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)

    if top == 10:
        random_baseline = proportions[0]
    else:
        random_baseline = proportions[1]

    return random_baseline


def predict_from_doc_ranking(top=10):
    """
    Calculate the PhD Ranking heuristic (i.e., predict 'high' if the PhD rank is 'high').

    :param top: the threshold for 'high' (int)
    :return: the PR-AUC of the heuristic
    """
    classes = [0, 1, 2]

    test_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_test.csv')
    labels = test_df['y']

    # Set up splits for binned rank
    conditions = [
        (test_df['doctoral_ranking'] >= 1) & (test_df['doctoral_ranking'] <= top),
        (test_df['doctoral_ranking'] >= top + 1) & (test_df['doctoral_ranking'] <= top + 10),
        (test_df['doctoral_ranking'] > top + 10)
    ]


    # Calculate binned ranking
    choices = [0, 1, 2]
    class_0_prauc = 0
    test_df['doc_y'] = np.select(conditions, choices, default=2)
    test_df['doc_y'] = test_df['doc_y'].astype(int)
    doc_ranking = test_df['doc_y']

    # Generate PR-AUC
    for target_class in classes:
        if target_class is not None:
            y_true = (labels == target_class).astype(int)
            y_pred = (doc_ranking == target_class).astype(int)

        pr_auc = average_precision_score(y_true, y_pred)

        if target_class == 0:
            class_0_prauc = pr_auc

    print(f'by PhD PR_AUC: {class_0_prauc}')
    return class_0_prauc


def predict_from_avg_neighbor_rank(top=10):
    """
    Calculate the by Average Neighbor heuristic (i.e., predict 'high' if the
    average rank of one's faculty neighbors is 'high').

    :param top: the threshold for 'high' (int)
    :return: the PR-AUC of the heuristic
    """

    faculty_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty.csv')
    test_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_test.csv', index_col='fac_idx')
    graph_data = np.load(DATA_DIR / 'graph_adjmats' / 'graph.npz', allow_pickle=True)
    adjs = graph_data['adjs']

    avg_neighbor_rankings = []

    for idx, row in test_df.iterrows():
        hire_year = row['join_year']
        graph_year = hire_year - 1

        # Skip if graph year is out of bounds
        if graph_year < 2010 or graph_year > 2020:
            avg_neighbor_rankings.append(np.nan)
            continue

        graph_idx = graph_year - 2010  # indexing from 0
        adj_matrix = adjs[graph_idx]

        # Find co-authors (non-zero entries in adjacency matrix row)
        coauthors = np.where(adj_matrix[idx] > 0)[0]

        neighbor_rankings = []
        for neighbor in coauthors:
            if faculty_df.loc[neighbor, f'fac_{graph_year}'] == 1:
                neighbor_rankings.append(faculty_df.loc[neighbor, 'ranking'])

        # Compute average if non-empty
        if neighbor_rankings:
            avg_neighbor_rankings.append(np.mean(neighbor_rankings))
        else:
            avg_neighbor_rankings.append(-1)

    test_df['avg_neighbor_ranking'] = avg_neighbor_rankings

    # Get neighbor ranks
    conditions = [
        (test_df['avg_neighbor_ranking'] >= 1) & (test_df['avg_neighbor_ranking'] <= top),
        (test_df['avg_neighbor_ranking'] >= top+1) & (test_df['avg_neighbor_ranking'] <= top+10),
        (test_df['avg_neighbor_ranking'] > top+10),
        (test_df['avg_neighbor_ranking'] == -1)
    ]

    choices = [0, 1, 2, 2]
    test_df['neighbor_y'] = np.select(conditions, choices, default=2)
    test_df['neighbor_y'] = test_df['neighbor_y'].astype(int)
    avg_neighbor_ranking = test_df['neighbor_y']

    test_df = pd.read_csv('../data/hiring/faculty_test.csv')
    labels = test_df['y']

    # Generate MCC
    classes = [0, 1, 2]
    class_0_prauc = 0
    for target_class in classes:
        if target_class is not None:
            y_true = (labels == target_class).astype(int)
            y_pred = (avg_neighbor_ranking == target_class).astype(int)

        pr_auc = average_precision_score(y_true, y_pred)

        if target_class == 0:
            class_0_prauc = pr_auc

    print(f'by Avg Co-author Rank MCC: {class_0_prauc}')
    return class_0_prauc


def main():
    """
    Generates heuristics for machine learning task.

    :return: None
    """
    get_random_guessing(top=10)
    predict_from_doc_ranking(top=10)
    predict_from_avg_neighbor_rank(top=10)


if __name__=='__main__':
    main()