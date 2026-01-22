"""
prep_for_gml.py

This script:
1) Packs yearly (author x author) adjacency matrices into a single .npz file.
2) Builds per-year feature snapshots from static and temporal columns.
3) Writes repeated label tensors over time for multiple top-k thresholds.
4) Creates train/val/test boolean masks aligned to the time axis.

Conventions
- Years = 2010...2020 (11 timesteps).
- Adjacency is (n_nodes x n_nodes), scipy.sparse CSR on disk; dense in memory when packing.
- Features are L2-normalized row-wise per timestep.
- Labels are repeated across time (no time-varying y).
- Masks: train/val are time-constant; test is time-specific for (2018, 2019, 2020 hires).

1/13/2026 — SD
"""

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import pandas as pd
import os


def prep_adj_mats(years, percent=None, rewire=None):
    """
    Load per-year author x author adjacency matrices and pack them into a single .npz.

    If `rewire`, loads from '../data/graph_adjmats_rewired/adjmat_{year}_{rewire}_{percent}.npz'
    and writes '../data/graph_adjmats_rewired/graph_{rewire}_{percent}.npz' with key 'adjs'.

    Otherwise, loads the original graphs from '../data/graph_adjmats/adjmat_{year}.npz'
    and writes '../data/graph_adjmats/graph.npz' with key 'adjs'.

    :param years: List of years (e.g., 2010...2020) in ascending order (Sequence[int])
    :param percent: Rewiring percentage (only used when rewire is given) (int)
    :param rewire: Flag that indicates using rewired graphs; otherwise original graphs.
    :return: None
    """
    adjmats = []
    for year in years:
        if rewire:
            fp = f'../data/graph_adjmats_rewired/adjmat_{year}_{rewire}_{percent}.npz'
            adj_csr = scipy.sparse.load_npz(fp)
        else:
            adj_csr = scipy.sparse.load_npz(f'../data/graph_adjmats/adjmat_{year}.npz')
        adj_dense = adj_csr.toarray()
        adjmats.append(adj_dense)

    if rewire:
        np.savez_compressed(
            f'../data/graph_adjmats_rewired/graph_{rewire}_{percent}.npz',
            adjs=adjmats,
        )
    else:
        np.savez_compressed(
            f'../data/graph_adjmats/graph.npz',
            adjs=adjmats,
        )


def load_features():
    """
    Read feature name lists from plain-text files (one name per line).

    :return: dictionary with features for the given experiment (Dict[str, List[str])
    """
    with open('../data/hiring/cv_features.txt', 'r') as f:
        cv_features = [line.strip() for line in f.readlines()]

    with open('../data/hiring/bibliometric_features.txt', 'r') as f:
        biblio_features = [line.strip() for line in f.readlines()]

    with open('../data/hiring/coauthorship_features.txt', 'r') as f:
        graph_features = [line.strip() for line in f.readlines()]

    cv_biblio_features = cv_features + biblio_features
    cv_graph_features = cv_features + graph_features
    biblio_graph_features = biblio_features + graph_features
    cv_biblio_graph_features = cv_features + biblio_features + graph_features

    feature_sets = {
        'cv': cv_features,
        'biblio': biblio_features,
        'graph': graph_features,
        'cv+biblio': cv_biblio_features,
        'cv+graph': cv_graph_features,
        'biblio+graph': biblio_graph_features,
        'cv+biblio+graph': cv_biblio_graph_features
    }

    return feature_sets


def build_feature_snapshots(selected_features, faculty_df, years):
    """
    Build per-year (time-indexed) feature matrices.

    Feature selection logic:
      - "Temporal" feature families are detected by the presence of columns that
        start with '<feat>_' in the faculty_df (e.g., 'papers_2015').
      - "Static" feature families are those without per-year suffix columns.

    For each year:
      - Collect all static columns for `selected_features`.
      - Collect temporal columns by formatting '<feat>_{year}' for each temporal feat.
      - hstack [static, temporal] to (n_nodes, n_features_this_year)
      - Normalize rows.

    :param selected_features: Feature families or column names. Temporal families must exist as '<feat>_{year}' (Sequence[str])
    :param faculty_df: Master faculty features table containing static and/or temporal columns (pd.DataFrame)
    :param years: Years to snapshot (e.g., 2010...2020) (Sequence[int])
    :return: A list of length len(years), each (n_nodes, d_t) float32 array (List[np.ndarray])
    """
    features = []
    static_features = []
    temporal_features = []

    # Heuristic: if any column starts with "<feat>_", we treat feat as temporal family
    for feat in selected_features:
        if any(faculty_df.columns.str.startswith(f"{feat}_")):

            # gender also has 'gender_string', so treat as static
            if feat != 'gender':
                temporal_features.append(feat)
            else:
                static_features.append(feat)
        else:
            # Treat as static column name
            static_features.append(feat)

    if static_features:
        static_matrix = faculty_df[static_features].values  # (n_nodes, n_static_features)

    for year in years:
        time_features = []

        if static_features:
            time_features.append(static_matrix)

        if temporal_features:
            temporal_cols = [f"{feat}_{year}" for feat in temporal_features]
            temporal_matrix = faculty_df[temporal_cols].values  # (n_nodes, n_temporal_features)
            time_features.append(temporal_matrix)

        # Stack static + temporal features horizontally
        time_array = np.hstack(time_features)  # (n_nodes, total_features_at_this_time)

        # Normalize and store
        time_array = torch.from_numpy(time_array).float()
        time_array = F.normalize(time_array, p=2.0, dim=1)
        features.append(time_array.numpy())

    return features


def prep_features(years, faculty_df):
    """
    Build and save feature snapshots for each predefined feature set.

    Writes:
      '../data/graph_adjmats/features_{feat_type}.npz' with key 'attmats'

    :param years: Years to snapshot (Sequence[int])
    :param faculty_df: Master faculty feature table (pd.DataFrame)
    :return: None
    """
    feature_sets = load_features()

    for feat_type, feats in feature_sets.items():
        features = build_feature_snapshots(feats, faculty_df, years)

        np.savez_compressed(
            f'../data/graph_adjmats/features_{feat_type}.npz',
            attmats=features
        )


def prep_labels(faculty_df):
    """
    Write labels repeated across time for multiple top-k thresholds.

    Expects columns 'y', 'y_20', 'y_30', 'y_40', 'y_50' in `faculty_df`.
    Produces (T, N) arrays with identical rows (T=11 time steps).

    Writes:
      '../data/graph_adjmats/labels_{k}.npz' with key 'Labels'

    :param faculty_df: Faculty table with y_k columns (pd.DataFrame)
    :return:None
    """
    top = [10, 20, 30, 40, 50]
    for t in top:
        if top == 10:
            labels = faculty_df['y'].values  # shape: (num_nodes,)
            labels = labels.astype(int)

            labels_over_time = np.tile(labels, (11, 1))  # shape: (11, num_nodes)

            os.makedirs('../data/graph_adjmats', exist_ok=True)
            np.savez_compressed('../data/graph_adjmats/labels.npz', Labels=labels_over_time)

        else:
            labels = faculty_df[f'y_{t}'].values   # shape: (num_nodes,)
            labels = labels.astype(int)

            labels_over_time = np.tile(labels, (11, 1))  # shape: (11, num_nodes)

            os.makedirs('../data/graph_adjmats', exist_ok=True)
            np.savez_compressed(f'../data/graph_adjmats/labels_{t}.npz', Labels=labels_over_time)


def prep_masks(window=3, num_nodes=4656):
    """
    Build train/val/test boolean masks aligned to 11 time steps (2010–2020).

    Train/val selection:
      - Find earliest test join_year (min over test set).
      - Allow train/val join_year in [earliest- window, ..., earliest - 1].

    Test selection:
      - For time t with year in {2017, 2018, 2019}, mark nodes whose join_year = year + 1
        (i.e., predict hires in 2018, 2019, 2020).

    Writes:
      '../data/graph_adjmats/train_mask_w{window}.npz' (Labels = (T,N) bool array)
      '../data/graph_adjmats/val_mask_w{window}.npz'
      '../data/graph_adjmats/test_mask_w{window}.npz'

    :param window: Backward window size to define train/val join_year range (int)
    :param num_nodes: Number of nodes/authors (columns of masks) (int)
    :return: None
    """
    train_df = pd.read_csv('../data/hiring/faculty_train.csv')
    val_df = pd.read_csv('../data/hiring/faculty_val.csv')
    test_df = pd.read_csv('../data/hiring/faculty_test.csv')

    earliest_test_year = test_df['join_year'].min()
    allowed_years = set(range(earliest_test_year - window, earliest_test_year))

    train_indices = train_df[train_df['join_year'].isin(allowed_years)]['fac_idx'].tolist()
    val_indices = val_df[val_df['join_year'].isin(allowed_years)]['fac_idx'].tolist()

    timesteps = 11  # 2010–2020
    years = list(range(2010, 2021))  # year corresponding to each timestep index

    # Initialize empty boolean masks
    train_mask = np.zeros((timesteps, num_nodes), dtype=bool)
    val_mask = np.zeros((timesteps, num_nodes), dtype=bool)
    test_mask = np.zeros((timesteps, num_nodes), dtype=bool)

    # Fill train/val masks: stays constant across all timesteps
    for t in range(timesteps):
        train_mask[t, train_indices] = True
        val_mask[t, val_indices] = True

    # Fill test mask: varies depending on year
    for t, year in enumerate(years):
        if year in [2017, 2018, 2019]:  # predict 2018, 2019, 2020 hires
            hires_this_year = test_df[test_df['join_year'] == (year + 1)]['fac_idx'].tolist()
            test_mask[t, hires_this_year] = True

    save_dir = '../data/graph_adjmats'
    os.makedirs(save_dir, exist_ok=True)

    np.savez_compressed(os.path.join(save_dir, f'train_mask_w{window}.npz'), Labels=train_mask)
    np.savez_compressed(os.path.join(save_dir, f'val_mask_w{window}.npz'), Labels=val_mask)
    np.savez_compressed(os.path.join(save_dir, f'test_mask_w{window}.npz'), Labels=test_mask)


def main():
    """
    Example pipeline:
          1) Pack (possibly rewired) adjacency sequences to .npz
          2) Build feature snapshots (commented by default)
          3) Build repeated labels (commented by default)
          4) Build train/val/test masks (commented by default)

    :return: None
    """
    years = list(range(2010, 2021))
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')

    # save graphs
    print('getting adjacency matrices')
    prep_adj_mats(years)

    # save feature matrices
    print('getting feature matrices')
    prep_features(years, faculty_df)

    # save labels
    print('getting labels')
    prep_labels(faculty_df)

    # save train/val/test masks
    print('getting masks')
    prep_masks(window=1)
    prep_masks(window=2)
    prep_masks(window=3)


if __name__=='__main__':
    main()