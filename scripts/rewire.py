"""
rewire_graphs.py

Cumulative degree-preserving rewiring of yearly co-authorship graphs via
double-edge swaps, saving both non-cumulative (already exists on disk) and
cumulative adjacency matrices for each rewiring percentage and repetition.

Behavior:
- For each repetition (rewire_iter) and for each 'percent' in [10,20,...,100],
  we apply an additional ~10% double-edge swaps to the *current* graphs
  (i.e., rewiring is cumulative across percent steps within a repetition).
- After each step, we save the *cumulative* (year-over-year) adjacency matrices
  for 2010..2020 into ../data/graph_adjmats_rewired/.

Notes:
- Uses a custom double-edge swap that preserves degree sequence and avoids
  parallel edges/self-loops; the graph is modified in-place.
- Assumes the presence of non-cumulative yearly adjacency matrices at:
    ../data/graph_adjmats/non_cumulative_adjmat_{YEAR}.npz
- Output files:
    ../data/graph_adjmats_rewired/adjmat_{YEAR}_{rewire_iter}_{percent}.npz
  where matrices are cumulative-over-years within that repetition/percent.

10/28/2025 - SD
"""

import os
import networkx as nx
import tqdm
import random
from scipy.sparse import csr_matrix, save_npz, load_npz


def double_edge_swap(G, nswap=1, max_tries=100, seed=None):
    """
    Perform degree-preserving rewiring by repeatedly attempting double-edge swaps.

    A double-edge swap removes edges (u, v) and (x, y), then adds (u, x) and (v, y),
    provided those edges do not already exist and do not create self-loops.

    :param G: Undirected simple graph. Modified in-place (nx.Graph)
    :param nswap: Number of *successful* swaps to perform (int)
    :param max_tries: Maximum number of *attempts* allowed to achieve 'nswap' successes (int)
    :param seed: Random seed. Used for degree-based sampling and endpoint selection.
    :return: The input graph `G` (modified in place) (nx.Graph)
    """

    if G.is_directed():
        raise nx.NetworkXError("double_edge_swap() not defined for directed graphs.")
    if nswap > max_tries:
        raise nx.NetworkXError("Number of swaps > number of tries allowed.")
    if len(G) < 4 or len(G.edges) < 2:
        raise nx.NetworkXError("Graph too small for edge swaps.")

    # Local RNG
    rng = random.Random(seed)

    # Precompute a degree-weighted sampling distribution over nodes
    keys, degrees = zip(*G.degree())
    cdf = nx.utils.cumulative_distribution(degrees)
    discrete_sequence = nx.utils.discrete_sequence

    n_attempts = 0
    swaps_done = 0

    while swaps_done < nswap:
        # Sample two (distinct) nodes, weighted by degree
        ui, xi = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue
        u, x = keys[ui], keys[xi]

        # Choose random neighbors for each endpoint
        v = rng.choice(list(G[u]))
        y = rng.choice(list(G[x]))

        # Avoid creating self-loops or trivial swaps
        if v == y or u == x or u == y or v == x:
            n_attempts += 1
            if n_attempts >= max_tries:
                raise nx.NetworkXAlgorithmError(
                    f"Exceeded {max_tries} attempts before achieving {nswap} swaps."
                )
            continue

        # Check that new edges don't already exist
        if (x not in G[u]) and (y not in G[v]):
            # Perform swap
            G.add_edge(u, x)
            G.add_edge(v, y)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swaps_done += 1

        n_attempts += 1
        if n_attempts >= max_tries:
            raise nx.NetworkXAlgorithmError(
                f"Exceeded {max_tries} attempts before achieving {nswap} swaps."
            )

    return G


def reform_adjmats(rewired_graphs, rewire_iter, percent, outdir):
    """
    Convert a list of yearly graphs into cumulative-over-years adjacency matrices
    and save each year as a sparse .npz.

    :param rewired_graphs: Graphs for years [2010..2020], already rewired for the current step
    :param rewire_iter: Repetition index of rewiring (e.g., 0..9)
    :param percent: Label for the cumulative rewiring step (10, 20, ..., 100)
    :param outdir: Directory to save cumulative adjacency matrices
    :return: None
    """
    os.makedirs(outdir, exist_ok=True)
    years = list(range(2010, 2021))
    adjmats = [nx.to_numpy_array(G) for G in rewired_graphs]

    prev = None
    for i, A in enumerate(adjmats):
        cumulative = A if prev is None else (A + prev)
        save_npz(
            os.path.join(outdir, f'adjmat_{years[i]}_{rewire_iter}_{percent}.npz'),
            csr_matrix(cumulative)
        )
        prev = cumulative


def rewire_cumulative(graphs, rewire_iter, percent_rewire_list, step_fraction=0.10, seed=None):
    """
    Rewire graphs cumulatively across increasing percentages.

    For each 'percent' in 'percent_rewire_list', perform an additional
    'step_fraction' * E double-edge swaps (E = current number of edges)
    on each graph, *starting from the previously rewired graph*.

    :param graphs: Starting graphs for years [2010..2020] (non-cumulative per-year state)
    :param rewire_iter: Repetition index (for multiple independent rewiring runs)
    :param percent_rewire_list: Labels for cumulative percentages (e.g., [10,20,...,100])
    :param step_fraction: Fraction of edges to swap per step. Default 0.10 (10%)
    :param seed: Base RNG seed for reproducibility. Each year graph uses a derived seed.
    :return: None
    """

    current = [G.copy() for G in graphs]  # Start from original graphs

    for percent in percent_rewire_list:
        rewired = []
        for year_idx, G in enumerate(tqdm.tqdm(current, desc=f"Rewire iter {rewire_iter}, {percent}%")):
            # Number of *successful* swaps to attempt this step (~10% of edges)
            nswap = max(1, int(step_fraction * G.number_of_edges()))

            # Derive a reproducible per-graph seed
            year_seed = None if seed is None else (seed + rewire_iter * 10_000 + year_idx * 1_000 + percent)
            G = double_edge_swap(G, nswap=nswap, max_tries=1_000_000, seed=year_seed)
            rewired.append(G)

        # Save cumulative-over-years adjacency matrices for this step
        outdir = f'../data/graph_adjmats_rewired/'
        reform_adjmats(rewired, rewire_iter, percent, outdir)

        # Carry forward for the next (more rewired) step
        current = rewired


def load_yearly_graphs( in_dir='../data/graph_adjmats', years=None):
    """
    Load non-cumulative yearly adjacency matrices (sparse .npz) and convert to graphs.

    :param in_dir: Directory containing non-cumulative adjacency files named 'non_cumulative_adjmat_{YEAR}.npz'
    :param years: Years to load
    :return: One simple undirected graph per year
    """
    if years is None:
        years = list(range(2010, 2021))

    graphs = []
    for year in years:
        path = os.path.join(in_dir, f"non_cumulative_adjmat_{year}.npz")
        A = load_npz(path).toarray()

        # Interpret any nonzero as an edge (simple, unweighted graph)
        G = nx.from_numpy_array((A > 0).astype(int))
        graphs.append(G)

    return graphs


def get_rewires(num_rewires=10, percent_rewire_list=None, base_in_dir='../data/graph_adjmats', seed=None):
    """
    Orchestrate multiple independent cumulative rewiring runs.

    :param num_rewires: Number of independent repetitions (different random seeds/paths)
    :param percent_rewire_list: Cumulative step labels to process; default [10,20,...,100]
    :param base_in_dir: Where to load non-cumulative yearly adjacencies from
    :param seed: Base RNG seed
    :return: None
    """
    if percent_rewire_list is None:
        percent_rewire_list = list(range(10, 101, 10))

    graphs = load_yearly_graphs(in_dir=base_in_dir)

    for rewire_iter in tqdm.tqdm(range(num_rewires), desc="Rewire repetitions"):
        rewire_cumulative(
            graphs=graphs,
            rewire_iter=rewire_iter,
            percent_rewire_list=percent_rewire_list,
            step_fraction=0.10,  # fixed +10% swaps per step
            seed=None if seed is None else (seed + 1_000_000 * rewire_iter)
        )


def main():
    """
    Rewire networks with degree-preserving rewiring.

    :return: None
    """

    get_rewires(
        num_rewires=10,
        percent_rewire_list=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        base_in_dir="../data/graph_adjmats",
    )


if __name__ == '__main__':
    main()