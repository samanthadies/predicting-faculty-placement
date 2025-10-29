"""
get_coauthorship_features.py

Compute graph-based features per year and append them to the faculty table.
For each adjacency matrix (author x author) in ../data/graph_adjmats/adjmat_{YEAR}.npz,
this script builds a NetworkX graph and computes:
- degree
- clustering coefficient
- PageRank (α = 0.85)
- betweenness centrality
- eigenvector centrality (with a safe fallback when it does not converge)

Each feature is stored as a new column on `faculty.csv` with the pattern
`{feature_name}_{year}`. Mapping from graph node to row is done by using
the DataFrame index (or an optional id column).

10/24/2025 — SD
"""

import os
from typing import Dict, Iterable, Mapping, Optional

import networkx as nx
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm


def load_graph_from_adj(path):
    """
    Load a scipy CSR adjacency from disk and convert to a NetworkX graph.
    Uses NetworkX's array→graph converter with integer node labels [0, n-1].

    :param path: file path to adjacency matrix (str)
    :return: G (nx.Graph)
    """

    adj_csr = sp.load_npz(path)
    G = nx.from_scipy_sparse_array(adj_csr)

    # Ensure we treat edges as undirected
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    return G


def compute_features(G):
    """
    Compute a suite of centrality/structure features on G.

    :param G: graph (nx.Graph)
    :return: dict of features (Dict[int, float])).
    """
    # Degree centrality here is raw degree (int), not normalized.
    deg = dict(G.degree())

    # Clustering coefficient (triangle density around a node)
    clustering = nx.clustering(G)

    # PageRank with damping factor alpha
    pagerank = nx.pagerank(G, alpha=0.85)

    # Betweenness (can be slow; exact computation)
    betweenness = nx.betweenness_centrality(G)

    # Eigenvector centrality may not converge on some graphs; fallback to zeros
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.NetworkXError:
        eigenvector = {n: 0.0 for n in G.nodes()}

    return {
        "degree": deg,
        "clustering": clustering,
        "pagerank": pagerank,
        "betweenness": betweenness,
        "eigenvector": eigenvector,
    }


def attach_yearly_features(faculty_df, features_by_year, years, node_id_col):
    """
    Merge per-year feature dictionaries into the faculty_df as new columns.

    :param faculty_df: Table of authors; must align with graph node ids (pd.DataFrame)
    :param features_by_year: Nested mapping of features computed on the yearly graphs
    :param years: Which years to add (Sequence[int])
    :param node_id_col: If provided, use this column’s values (e.g., 'fac_idx') to map features (str)
    :return: Copy of faculty_df with additional columns `{feature_name}_{year}' (pd.DataFrame)
    """

    df = faculty_df.copy()

    # Series of node ids used to look up feature values
    if node_id_col is None:
        node_ids = df.index
    else:
        if node_id_col not in df.columns:
            raise KeyError(f"node_id_col '{node_id_col}' not found in DataFrame.")
        node_ids = df[node_id_col]

    for year in years:
        yearly = features_by_year[year]  # {'degree': {id: val}, ...}
        for feat_name, mapping in yearly.items():
            col = f"{feat_name}_{year}"
            # Map each node_id to its value; missing ids become NaN
            df[col] = pd.Series(node_ids).map(mapping).to_numpy()

    return df


def process_years(years, adjmat_dir):
    """
    Load each year's adjacency, build a graph, compute features.

    :param years: list of years (List[int])
    :param adjmat_dir: file path to adjmat (str)
    :return: dictionary of features
    """

    results: Dict[int, Dict[str, Mapping[int, float]]] = {}

    for year in tqdm(list(years), desc="Processing years"):
        adj_fp = os.path.join(adjmat_dir, f"adjmat_{year}.npz")
        G = load_graph_from_adj(adj_fp)
        results[year] = compute_features(G)

    return results


def main():
    """
    Generates graph-based features for the co-authorship network snapshots.

    :return: None
    """

    years = list(range(2010, 2021))
    adjmat_dir = "../data/graph_adjmats"
    faculty_fp = "../data/hiring/faculty.csv"
    node_id_col = None  # set to 'fac_idx' if your faculty table has an explicit id column

    # Load faculty and compute graph features
    faculty_df = pd.read_csv(faculty_fp)
    features_by_year = process_years(years, adjmat_dir)

    # Attach features and save
    faculty_out = attach_yearly_features(faculty_df, features_by_year, years, node_id_col=node_id_col)
    faculty_out.to_csv(faculty_fp, index=False)
    print(f"Saved updated faculty file with features to: {faculty_fp}")


if __name__ == "__main__":
    main()