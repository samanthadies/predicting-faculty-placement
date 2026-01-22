"""
graph_ml.py

Graph machine learning modeling pipeline for faculty placement prediction. This script
1) Trains static graph models (with hyperparameter search)
2) Trains temporal graph models (with hyperparameter search)
3) Allows for different definitions of high-rank
4) Saves the best models and summaries of the results
5) Runs experiments on rewired graphs
6) Saves a summary dataframe of rewired experiments

Inputs:
data/
    graph.npz                                                       # {'adjs': List[np.ndarray]} - per-year adjacencies
    labels.npz OR labels_{top}.npz                                  # {'Labels': np.ndarray[t, n]} - per-year labels
    features_{feature}.npz                                          # {'attmats': List[np.ndarray]} - optional per-year node features
    train_mask_w{w}.npz, val_mask_w{w}.npz, test_mask_w{w}.npz      # {'Labels': np.ndarray[t, n]}
    graph_{iter}_{pct_rewire}.npz                                   # {'adjs': List[np.ndarray]} - per-year rewired adjacencies

Outputs:
output/models_{feature}/y_{top}/{model}/
    sweep/config_sweep_{idx}.json           # one per sweep index
    best_configs.csv                        # selected best configs (one row per (model,feature,class,weighted,top))
    repeat/run_{k}.csv                      # metrics per repeat
    rewire/run_iter{iter}_pct{pct}.csv      # rewired results (one per iter + pct combo)
    analysis/aggregates.csv                 # grouped means/stds
outputs/y_{top}/
    gml_aggregates.csv                      # aggregated GML results
    gml_aggregates_ranked_by_mcc.csv        # results ranked by MCC
    gml_aggregates_ranked_by_pr_auc.csv     # results ranked by PR-AUC
    gml_aggregates_wide.csv                 # pivoted table with aggregated results
    gml_combined_runs.csv                   # run-level results
rewired_summary.csv                         # run-level rewiring results

1/22/2026 - SD
"""

import sys, json, os, itertools, copy
import re
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    matthews_corrcoef, precision_recall_fscore_support,
    average_precision_score, accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.RecurrentGCN import RecurrentGCN


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


MODEL_REGISTRY = {
    "GCN": {
        "kind": "static",
        "class": GCN,
        "grid": {
            "hidden_channels": [[64], [128], [256], [512], [1024], [2048],
                                [128, 64], [256, 128], [512, 256], [1024, 512], [2048, 1024]],
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    },
    "GraphSAGE": {
        "kind": "static",
        "class": GraphSAGE,
        "grid": {
            "hidden_channels": [[64], [128], [256], [512], [1024], [2048],
                                [128, 64], [256, 128], [512, 256], [1024, 512], [2048, 1024]],
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    },
    "GAT": {
        "kind": "static",
        "class": GAT,
        "grid": {
            "hidden_channels": [[64], [128], [256], [512], [1024], [2048],
                                [128, 64], [256, 128], [512, 256], [1024, 512], [2048, 1024]],
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
    },
    "GConvGRU": {
        "kind": "temporal",
        "class": RecurrentGCN,
        "grid": {
            "hidden_channels": [[64], [128], [256], [512], [1024], [2048],
                                [128, 64], [256, 128], [512, 256], [1024, 512], [2048, 1024]],
            "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
            "K": 3,
        },
    },
}


def ensure_dir(p):
    """
    Make sure directories exist.

    :param p: filepath of directory
    :return: None
    """
    Path(p).mkdir(parents=True, exist_ok=True)


def data_paths(data_dir, feature, window, top, graph_fp_override=None):
    """
    Creates a dictionary of data filepaths.

    :param data_dir: Directory storing data
    :param feature: Feature types
    :param window: Sliding window
    :param top: What y's to consider as High-rank
    :param graph_fp_override: If provided, use this graph fp instead of data_dir/graph.npz
    :return: Dictionary with data filepaths.
    """
    labels_fp = os.path.join(data_dir, 'labels.npz' if top == 10 else f'labels_{top}.npz')
    features_fp = None if feature == 'none' else os.path.join(data_dir, f'features_{feature}.npz')

    graph_fp = graph_fp_override if graph_fp_override is not None else os.path.join(data_dir, 'graph.npz')

    return {
        'graph_fp':   graph_fp,
        'labels_fp':  labels_fp,
        'features_fp': features_fp,
        'train_mask_fp': os.path.join(data_dir, f'train_mask_w{window}.npz'),
        'val_mask_fp':   os.path.join(data_dir, f'val_mask_w{window}.npz'),
        'test_mask_fp':  os.path.join(data_dir, f'test_mask_w{window}.npz'),
    }


def load_graph_npz(graph_fp):
    """
    Load graph object. Expects a dict {'adjs': list of (n x n) numpy arrays}, one per time.

    :param graph_fp: filepath
    :return: list of adjacency matrices.
    """

    obj = np.load(graph_fp, allow_pickle=True)
    return list(obj['adjs'])


def load_features_npz(features_fp=None, T=11, n=4656):
    """
    Load features.

    :param features_fp: filepath to features
    :param T: number of time steps
    :param n: number of nodes
    :return: List of feature matrices
    """

    if features_fp is None:
        return [np.ones((n, 1), dtype=float) for _ in range(T)]

    obj = np.load(features_fp, allow_pickle=True)
    feats = list(obj['attmats'])
    out = []
    for t in range(T):
        X = feats[t].astype(float)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        out.append(X)

    return out


def load_labels_npz(labels_fp):
    """
    Loads labels*.npz. Expects a dict {'Labels': array of shape (T, n)} (int class labels)

    :param labels_fp: filepath to features
    :return: array of features
    """

    return np.load(labels_fp, allow_pickle=True)['Labels'].astype(int)


def load_masks(mask_fp):
    """
    Loads node masks.

    :param mask_fp: filepath to masks.
    :return: node masks
    """
    return np.load(mask_fp, allow_pickle=True)['Labels'].astype(bool)


def transform_data(model_name, graph_fp, features_fp, labels_fp, train_mask_fp, val_mask_fp, test_mask_fp, weighted,
                   target_class=0, top=None):
    """
    Load and prepare data.

    :param model_name: name of model (to differentiate between GConvGRU and static models)
    :param graph_fp: filepath to graph adjacency matrices
    :param features_fp: filepath to features
    :param labels_fp: filepath to labels
    :param train_mask_fp: filepath to train mask
    :param val_mask_fp: filepath to val mask
    :param test_mask_fp: filepath to test mask
    :param weighted: whether the graph is weighted
    :param target_class: the target class
    :param top: what y's to consider 'high-rank'
    :return: data object
    """
    # get graph Adjmats
    adjs = load_graph_npz(graph_fp)
    T = len(adjs)
    n = adjs[0].shape[0]

    # Get Feature matrices
    feats = load_features_npz(features_fp, T, n)

    # Get labels
    labels = load_labels_npz(labels_fp)  # (T, n)
    if top == 10:  # Already done if top != 10
        labels = (labels == target_class).astype(int)

    # Get node masks
    train_mask = load_masks(train_mask_fp)  # (T, n) bool
    val_mask = load_masks(val_mask_fp)
    test_mask = load_masks(test_mask_fp)

    # process graphs into edge lists and weights
    data = []
    edges = []
    edge_weights = []
    for t in range(T):
        A = adjs[t]
        src, dst = np.nonzero(A)
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edges.append(edge_index)
        if weighted:
            edge_weight = torch.tensor(A[src, dst].astype(float), dtype=torch.float)
        else:
            edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)
        edge_weights.append(edge_weight)

        if model_name == 'GConvGRU':
            # construct data object for temporal graphs
            snapshot = Data(
                x=torch.tensor(feats[t], dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_weight=torch.tensor(edge_weight, dtype=torch.float),
                y=torch.tensor(labels[t], dtype=torch.long),
                train_mask=torch.tensor(train_mask[t], dtype=torch.bool),
                val_mask=torch.tensor(val_mask[t], dtype=torch.bool),
                test_mask=torch.tensor(test_mask[t], dtype=torch.bool)
            )
            data.append(snapshot)

    if data:
        return data

    else:
        # Create dataset for static graphs
        dataset = DynamicGraphTemporalSignal(
            edge_indices=edges,
            edge_weights=edge_weights,
            features=feats,
            targets=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        for t, snapshot in enumerate(dataset):
            snapshot.edge_weight = torch.tensor(edge_weights[t], dtype=torch.float)

        return dataset


def get_snapshot_for_hire_year(hire_year, dataset, graph_years):
    """
    Returns graph snapshot for the specific hire year (for static models).

    :param hire_year: hire year
    :param dataset: graph dataset
    :param graph_years: years in the graph to consider
    :return: graph snapshot
    """
    graph_year = hire_year - 1
    if graph_year not in graph_years:
        raise ValueError(f"Graph data for year {graph_year} (one year before {hire_year}) not available.")

    graph_idx = graph_years.index(graph_year)
    snapshot = dataset[graph_idx]

    snapshot.train_mask = torch.tensor(dataset.train_mask[graph_idx], dtype=torch.bool)
    snapshot.val_mask = torch.tensor(dataset.val_mask[graph_idx], dtype=torch.bool)
    snapshot.test_mask = torch.tensor(dataset.test_mask[graph_idx], dtype=torch.bool)

    return snapshot


def train_static(graph, model, epochs=500, patience=10, save_path=None):
    """
    Train a static GNN (GCN/GAT/GraphSAGE) on a single snapshot using masks with
    early-stops on validation loss.

    :param graph: graph snapshot
    :param model: model
    :param epochs: number of epochs to train for
    :param patience: patience to wait
    :param save_path: output path to save best model
    :return: trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_model_state = None
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x, graph.edge_index)

        cls_weights = class_weight.compute_class_weight('balanced', classes=np.unique(graph.y[graph.train_mask]),
                                                        y=graph.y[graph.train_mask].numpy())
        cls_weights = torch.tensor(cls_weights, dtype=torch.float, device=graph.x.device)
        criterion = torch.nn.CrossEntropyLoss(weight=cls_weights)

        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        val_loss = criterion(out[graph.val_mask], graph.y[graph.val_mask])

        loss.backward()
        optimizer.step()

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            if save_path is not None:
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_static(model, graph):
    """
    Evaluate trained model.

    :param model: model
    :param graph: graph snapshot
    :return: evaluation metrics
    """
    model.eval()
    out = model(graph.x, graph.edge_index)

    # Get model output
    probs = F.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)
    y_true = graph.y[graph.test_mask]
    y_pred = preds[graph.test_mask]
    y_probs = probs[graph.test_mask]

    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_probs = y_probs.cpu().detach().numpy()

    # Calculate metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    pr_auc = average_precision_score(y_true, y_probs)

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    return mcc, precision, recall, f1, cm, pr_auc, y_true, y_pred, y_probs


def train_temporal(dataset, model, device='cpu', epochs=200):
    """
    Train a temporal GNN (RecurrentGCN/GConvGRU).

    :param dataset: temporal graph dataset
    :param model: model
    :param device: training device
    :param epochs: epochs to train for
    :return: trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, val_loss = 0, 0
        H = None  # hidden memory passed through time, starts at None

        for snapshot in dataset:
            # load snapshot information
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_weight.to(device)
            y = snapshot.y.to(device)
            train_mask = snapshot.train_mask.to(device)
            val_mask = snapshot.val_mask.to(device)

            # set up class weights
            class_weights = torch.tensor(
                compute_class_weight('balanced', classes=np.unique(y[train_mask].cpu().numpy()),
                                     y=y[train_mask].cpu().numpy()), dtype=torch.float, device=device
            )
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            # generate logits
            out, H = model(x, edge_index, edge_weight, H=H)
            loss = criterion(out[train_mask], y[train_mask])
            val = criterion(out[val_mask], y[val_mask])

            # back prop through time
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            val_loss += val.item()

            if H is not None:
                H = H.detach()  # Prevent gradient accumulation

        # look for best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_temporal(model, dataset, test_year_indices):
    """
    Evaluate trained model.

    :param model: model
    :param dataset: graph dataset
    :param test_year_indices: test years to evaluate
    :return: Evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    all_y_true, all_y_pred, all_y_probs = [], [], []
    H = None

    # Generate model predictions for the relevant snapshots
    with torch.no_grad():
        for t, snapshot in enumerate(dataset):
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_weight.to(device)

            out, H = model(x, edge_index, edge_weight, H=H)

            probs = F.softmax(out, dim=1)[:, 1].detach().cpu()
            preds = out.argmax(dim=1).detach().cpu()
            mask = snapshot.test_mask.cpu()
            y_cpu = snapshot.y.cpu()

            if t in test_year_indices:
                all_y_true.append(y_cpu[mask].numpy())
                all_y_pred.append(preds[mask].numpy())
                all_y_probs.append(probs[mask].numpy())

    if not all_y_true:
        raise ValueError("No test examples found — check your test_year_indices or test masks.")

    # Concatenate across snapshots
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    y_probs = np.concatenate(all_y_probs)

    # Calculate metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    pr_auc = average_precision_score(y_true, y_probs)
    accuracy = (y_true == y_pred).mean()

    return mcc, precision, recall, f1, pr_auc, accuracy


def parallelized_hyperparameter_sweep(model_name, feature_set, top, sweep_idx):
    """
    Runs a hyperparameter tuning step based on command-line input (only one at a time for parallelization).

    :param model_name: name of model (key in MODEL_REGISTRY)
    :param feature_set: feature set combo (graph, cv, biblio, graph+cv, graph+biblio, cv+biblio, graph+cv+biblio)
    :param top: what y's to consider as High-rank
    :return: None
    """
    data_dir = DATA_DIR
    out_dir = OUTPUT_DIR / f'models_{feature_set}'
    ensure_dir(out_dir)

    # Define the sweep space
    windows = [1, 2, 3]
    weighted_options = [True, False]
    hidden_channels_options = MODEL_REGISTRY[model_name]['grid']['hidden_channels']
    dropout_options = MODEL_REGISTRY[model_name]['grid']['dropout']

    # set up the sweep
    sweep = list(itertools.product(windows, weighted_options, hidden_channels_options, dropout_options))
    if sweep_idx >= len(sweep):
        print(f"Error: sweep_idx {sweep_idx} out of range (total {len(sweep)} configs)")
        return

    window, weighted, hidden_channels, dropout = sweep[sweep_idx]
    print(
        f"\n=== Running sweep_idx={sweep_idx}: window={window}, weighted={weighted}, hidden={hidden_channels}, dropout={dropout} ===")

    # Data paths
    paths = data_paths(data_dir, feature_set, window, top)

    # Output paths
    out_dir = out_dir / f"y_{top}" / model_name / "sweep"
    ensure_dir(out_dir)

    # Targets
    target_class = 0

    # Years
    graph_years = list(range(2010, 2021))
    test_years = [2018, 2019, 2020]

    # Determine if static or temporal, and run corresponding hyperparameter run
    model_kind = MODEL_REGISTRY[model_name]['kind']

    if model_kind == 'static':
        # Get dataset object
        dataset = transform_data(
            model_name=model_name, graph_fp=paths['graph_fp'], features_fp=paths['features_fp'],
            labels_fp=paths['labels_fp'], train_mask_fp=paths['train_mask_fp'],
            val_mask_fp=paths['val_mask_fp'], test_mask_fp=paths['test_mask_fp'],
            weighted=weighted, target_class=target_class, top=top
        )

        all_y_true, all_y_pred, all_y_probs = [], [], []

        for year in test_years:
            snapshot = get_snapshot_for_hire_year(year, dataset, graph_years)
            num_features = snapshot.x.shape[1]

            model_cls = MODEL_REGISTRY[model_name]['class']
            model = model_cls(num_features=num_features, hidden_channels=hidden_channels, num_classes=2, dropout=dropout)

            # Train
            trained_model = train_static(snapshot, model, epochs=500, patience=50)

            # Evaluate
            mcc, precision, recall, f1, cm, pr_auc, y_true, y_pred, y_probs = evaluate_static(trained_model, snapshot)

            print(f'\t\tYear {year} | MCC: {mcc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f}')

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_probs.append(y_probs)

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        all_y_probs = np.concatenate(all_y_probs)

        pr_auc_class = average_precision_score(all_y_true, all_y_probs)
        mcc_class = matthews_corrcoef(all_y_true, all_y_pred)

        per_class_pr_auc = [pr_auc_class]
        per_class_mcc = [mcc_class]
        avg_pr_auc = float(np.mean(per_class_pr_auc))
        avg_mcc = float(np.mean(per_class_mcc))

        hparams = {'hidden_channels': hidden_channels, 'dropout': dropout}

    else:
        # model kind is temporal

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        test_year_indices = [graph_years.index(y) for y in test_years]

        per_class_pr_auc = []

        temporal_data = transform_data(
            model_name='GConvGRU', graph_fp=paths['graph_fp'], features_fp=paths['features_fp'],
            labels_fp=paths['labels_fp'], train_mask_fp=paths['train_mask_fp'], val_mask_fp=paths['val_mask_fp'],
            test_mask_fp=paths['test_mask_fp'], weighted=weighted,
            target_class=0, top=top
        )

        in_channels = temporal_data[0].x.shape[1]
        K = MODEL_REGISTRY[model_name]['grid']['K']

        model_cls = MODEL_REGISTRY[model_name]['class']
        model = model_cls(in_channels=in_channels, out_channels=2, K=K, hidden_channels=hidden_channels).to(device)

        # Move snapshots to device
        for snapshot in temporal_data:
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_weight = snapshot.edge_weight.to(device)
            snapshot.y = snapshot.y.to(device)
            snapshot.train_mask = snapshot.train_mask.to(device)
            snapshot.val_mask = snapshot.val_mask.to(device)
            snapshot.test_mask = snapshot.test_mask.to(device)

        # Train
        train_temporal_data = [snap for i, snap in enumerate(temporal_data) if i not in test_year_indices]
        trained_model = train_temporal(train_temporal_data, model, device=device, epochs=500)

        # Evaluate
        mcc, precision, recall, f1, pr_auc, accuracy = evaluate_temporal(trained_model, temporal_data, test_year_indices)

        print(f'PR-AUC: {pr_auc:.3f} | MCC: {mcc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')

        per_class_pr_auc = [float(pr_auc)]
        per_class_mcc = [float(mcc)]
        avg_pr_auc = float(np.mean(per_class_pr_auc))
        avg_mcc = float(np.mean(per_class_mcc))

        hparams = {'hidden_channels': hidden_channels, 'dropout': dropout, 'K': K}

    # save results
    config_save = {
        'window': window,
        'weighted': weighted,
        'hparams': hparams,
        'avg_pr_auc': avg_pr_auc,
        'per_class_pr_auc': [float(x) for x in per_class_pr_auc],
        'avg_mcc': avg_mcc,
        'per_class_mcc': [float(x) for x in per_class_mcc],
    }
    with open(out_dir / f'config_sweep_{sweep_idx}.json', 'w') as f:
        json.dump(config_save, f, indent=4)


def find_best_hyperparams(model_name, feature_set, top):
    """
    Read all hyperparam sweep JSONs for (model, feature, top) and pick the best by avg_pr_auc (and tie-break by avg_mcc).
    Write a single CSV `best_configs.csv`.

    :param model_name: name of model (key in MODEL_REGISTRY)
    :param feature_set: feature set combo (graph, cv, biblio, graph+cv, graph+biblio, cv+biblio, graph+cv+biblio)
    :param top: what y's to consider as High-rank
    :return: None
    """
    out_dir = OUTPUT_DIR / f'models_{feature_set}'
    sweep_dir = out_dir / f"y_{top}" / model_name / "sweep"

    rows = []
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(f"No sweep dir: {sweep_dir}")

    for fname in os.listdir(sweep_dir):
        if not fname.startswith("config_sweep_") or not fname.endswith(".json"):
            continue
        with open(sweep_dir / fname, "r") as f:
            cfg = json.load(f)
        rows.append(cfg)

    if not rows:
        raise RuntimeError(f"No sweep JSONs found in {sweep_dir}")

    # Pick best by avg_pr_auc, tie-break avg_mcc
    rows.sort(key=lambda r: (r.get("avg_pr_auc", 0.0), r.get("avg_mcc", 0.0)), reverse=True)
    best = rows[0]

    out_dir = out_dir / f"y_{top}" / model_name
    ensure_dir(out_dir)
    best_csv = out_dir / "best_configs.csv"

    # save a dataframe with the best result per model+feature+top combination
    df = pd.DataFrame([{
        "model_name": model_name,
        "feature_set": feature_set,
        "target_class": 0,
        "top": top,
        "window": best["window"],
        "weighted": best["weighted"],
        "hidden_channels": json.dumps(best["hparams"]["hidden_channels"]),
        "dropout": best["hparams"]["dropout"],
        "K": best["hparams"].get("K", None),
        "avg_pr_auc": best["avg_pr_auc"],
        "avg_mcc": best["avg_mcc"],
        "per_class_pr_auc": json.dumps(best["per_class_pr_auc"]),
        "per_class_mcc": json.dumps(best["per_class_mcc"]),
    }])
    df.to_csv(best_csv, index=False)


def repeat_best_runs(model_name, feature_set, top, n_runs):
    """
    Re-train best config N times; save per-run metrics CSV + best model state_dict per run.

    :param model_name: name of model (key in MODEL_REGISTRY)
    :param feature_set: feature set combo (graph, cv, biblio, graph+cv, graph+biblio, cv+biblio, graph+cv+biblio)
    :param top: what y's to consider as High-rank
    :param n_runs: number of repeated runs
    :return: None
    """

    data_dir = DATA_DIR
    out_dir = OUTPUT_DIR / f'models_{feature_set}'
    best_csv = out_dir / f"y_{top}" / model_name / "best_configs.csv"

    if not os.path.isfile(best_csv):
        raise FileNotFoundError(f"Missing best CSV: {best_csv}. Run best_hyperparams experiment first.")

    # load hyperparameters from best run
    row = pd.read_csv(best_csv).iloc[0]
    hidden_channels = json.loads(row["hidden_channels"])
    dropout = float(row["dropout"])
    K = int(row["K"]) if not pd.isna(row["K"]) else 3
    window = int(row["window"])
    weighted = bool(row["weighted"])

    # Filepath to save
    base_out = out_dir / f"y_{top}" / model_name / "repeat"
    ensure_dir(base_out)

    # Data
    paths = data_paths(data_dir, feature_set, window, top)
    graph_years = list(range(2010, 2021))
    test_years = [2018, 2019, 2020]

    # Targets
    target_class = 0

    # Determine if static or temporal, and run corresponding hyperparameter run
    model_kind = MODEL_REGISTRY[model_name]['kind']

    for run in range(n_runs):
        if model_kind == 'static':
            # Get dataset object
            dataset = transform_data(
                model_name=model_name, graph_fp=paths['graph_fp'], features_fp=paths['features_fp'],
                labels_fp=paths['labels_fp'], train_mask_fp=paths['train_mask_fp'],
                val_mask_fp=paths['val_mask_fp'], test_mask_fp=paths['test_mask_fp'],
                weighted=weighted, target_class=target_class, top=top
            )

            all_y_true, all_y_pred, all_y_probs = [], [], []

            for year in test_years:
                snapshot = get_snapshot_for_hire_year(year, dataset, graph_years)
                num_features = snapshot.x.shape[1]

                model_cls = MODEL_REGISTRY[model_name]['class']
                model = model_cls(num_features=num_features, hidden_channels=hidden_channels, num_classes=2,
                                  dropout=dropout)

                # Train
                trained_model = train_static(snapshot, model, epochs=500, patience=500)
                torch.save(model.state_dict(), base_out / f"best_model_year{year}_run{run}.pt")

                # Evaluate
                mcc, precision, recall, f1, cm, pr_auc, y_true, y_pred, y_probs = evaluate_static(trained_model, snapshot)

                print(
                    f'\t\tYear {year} | MCC: {mcc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f}')

                all_y_true.append(y_true)
                all_y_pred.append(y_pred)
                all_y_probs.append(y_probs)

            all_y_true = np.concatenate(all_y_true)
            all_y_pred = np.concatenate(all_y_pred)
            all_y_probs = np.concatenate(all_y_probs)

            # save a results dataframe per run
            metrics = dict(
                accuracy=accuracy_score(all_y_true, all_y_pred),
                precision=precision_recall_fscore_support(all_y_true, all_y_pred, average='binary')[0],
                recall=precision_recall_fscore_support(all_y_true, all_y_pred, average='binary')[1],
                f1=precision_recall_fscore_support(all_y_true, all_y_pred, average='binary')[2],
                pr_auc=average_precision_score(all_y_true, all_y_probs),
                mcc=matthews_corrcoef(all_y_true, all_y_pred),
                model_name=model_name,
                feature_set=f'{feature_set}+graph',
                run=run,
                top=top,
                weighted=weighted,
                target_class=0,
            )
            pd.DataFrame([metrics]).to_csv(base_out / f"run_{run}.csv", index=False)
            print(f"[REPEAT] {model_name} {feature_set} run {run}: MCC={metrics['mcc']:.3f} PR-AUC={metrics['pr_auc']:.3f}")

        else:
            # model is temporal
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            test_year_indices = [graph_years.index(y) for y in test_years]

            temporal_data = transform_data(
                model_name='GConvGRU', graph_fp=paths['graph_fp'], features_fp=paths['features_fp'],
                labels_fp=paths['labels_fp'], train_mask_fp=paths['train_mask_fp'], val_mask_fp=paths['val_mask_fp'],
                test_mask_fp=paths['test_mask_fp'], weighted=weighted,
                target_class=0, top=top
            )

            in_channels = temporal_data[0].x.shape[1]

            model_cls = MODEL_REGISTRY[model_name]['class']
            model = model_cls(in_channels=in_channels, out_channels=2, K=K,
                              hidden_channels=hidden_channels).to(device)

            # Move snapshots to device
            for snapshot in temporal_data:
                snapshot.x = snapshot.x.to(device)
                snapshot.edge_index = snapshot.edge_index.to(device)
                snapshot.edge_weight = snapshot.edge_weight.to(device)
                snapshot.y = snapshot.y.to(device)
                snapshot.train_mask = snapshot.train_mask.to(device)
                snapshot.val_mask = snapshot.val_mask.to(device)
                snapshot.test_mask = snapshot.test_mask.to(device)

            # Train
            train_temporal_data = [snap for i, snap in enumerate(temporal_data) if i not in test_year_indices]
            trained_model = train_temporal(train_temporal_data, model, device=device, epochs=500)
            torch.save(model.state_dict(), base_out / f"best_model_run{run}.pt")

            # Evaluate
            mcc, precision, recall, f1, pr_auc, accuracy = evaluate_temporal(trained_model, temporal_data,
                                                                             test_year_indices)

            print(f'PR-AUC: {pr_auc:.3f} | MCC: {mcc:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}')

            # save a results dataframe per run
            metrics = dict(
                accuracy=accuracy, precision=precision, recall=recall, f1=f1, pr_auc=pr_auc, mcc=mcc,
                model_name=model_name, feature_set=f'{feature_set}+graph', run=run, top=top,
                weighted=weighted, target_class=0,
            )
            pd.DataFrame([metrics]).to_csv(base_out / f"run_{run}.csv", index=False)
            print(f"[REPEAT] {model_name} {feature_set} run {run}: MCC={metrics['mcc']:.3f} PR-AUC={metrics['pr_auc']:.3f}")


def analyze_repeated_runs_for_top(top):
    """
    Aggregate all repeat runs across *all* models and features for a given 'top'.
    Outputs live under: output/gml/y_{top}/analysis/
    Produces:
      - combined_runs.csv                (all runs long-form)
      - aggregates.csv                   (mean/std by model, feature, top, weighted, target_classes)
      - aggregates_wide.csv              (features as columns; per model/top/weighted/target_classes)
      - aggregates_ranked_by_pr_auc.csv  (rank features within each model/top/weighted/target_classes)
      - aggregates_ranked_by_mcc.csv     (same, ranked by MCC)

    :param top: what y's to consider high-rank (int)
    :return: None
    """
    rows = []

    feature_roots = sorted(glob(str(OUTPUT_DIR / "models_*")))

    for feat_root in feature_roots:
        feat_root = Path(feat_root)
        feature = feat_root.name.split("models_", 1)[-1]

        y_dir = feat_root / f"y_{top}"
        if not y_dir.is_dir():
            continue

        for model in os.listdir(y_dir):

            base = y_dir / model / "repeat"
            if not base.is_dir():
                continue

            for fname in os.listdir(base):
                if fname.startswith("run_") and fname.endswith(".csv"):
                    fp = base / fname
                    try:
                        df = pd.read_csv(fp)
                    except Exception:
                        continue

                    # Ensure required identifiers are present and correct
                    df["model"] = model
                    df["feature"] = feature
                    df["top"] = int(top)
                    rows.append(df)

    if not rows:
        raise RuntimeError(f"No repeat CSVs found for top={top} under output/models_*/...")

    df = pd.concat(rows, ignore_index=True)

    # Output root
    out_root = OUTPUT_DIR / f"y_{top}"
    ensure_dir(out_root)

    # Save raw combined runs for transparency
    df.to_csv(out_root / "gml_combined_runs.csv", index=False)

    # Aggregates
    agg = (
        df.groupby(["model", "feature", "top", "weighted", "target_classes"])
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten MultiIndex columns
    agg.columns = [
        "_".join([c for c in col if c]) if isinstance(col, tuple) else col
        for col in agg.columns.values
    ]
    agg.to_csv(out_root / "gml_aggregates.csv", index=False)

    # Wide pivot (features as columns) for at-a-glance comparison per (model, top, weighted, target_classes)
    metric_cols = [c for c in agg.columns if c.endswith("_mean") or c.endswith("_std")]
    id_cols = ["model", "top", "weighted", "target_classes", "feature"]
    if all(c in agg.columns for c in id_cols):
        values = [m for m in metric_cols if m in (
            "pr_auc_mean", "mcc_mean", "precision_mean", "recall_mean", "f1_mean", "accuracy_mean"
        )]
        if values:
            wide = (
                agg.pivot_table(
                    index=["model", "top", "weighted", "target_classes"],
                    columns="feature",
                    values=values,
                    aggfunc="first",
                ).reset_index()
            )
            wide.columns = [
                "__".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
                for col in wide.columns
            ]
            wide.to_csv(out_root / "gml_aggregates_wide.csv", index=False)

    # Ranked tables (per model/top/weighted/target_classes)
    if "pr_auc_mean" in agg.columns:
        ranked_auc = agg.sort_values("pr_auc_mean", ascending=False)
        ranked_auc.to_csv(out_root / "gml_aggregates_ranked_by_pr_auc.csv", index=False)

        print(ranked_auc[['model', 'feature', 'pr_auc_mean', 'pr_auc_std']])

    if "mcc_mean" in agg.columns:
        ranked_mcc = agg.sort_values("mcc_mean", ascending=False)
        ranked_mcc.to_csv(out_root / "gml_aggregates_ranked_by_mcc.csv", index=False)


def rewire(model_name, feature_set, top, pct_rewired, iteration):
    """
    Train on a rewired graph.

    :param model_name: name of model
    :param feature_set: feature set combo
    :param top: what y's to consider as High-rank
    :param pct_rewired: rewiring percentage
    :param iteration: rewiring iteration
    :return: None
    """
    data_dir = DATA_DIR
    out_dir = OUTPUT_DIR / f"models_{feature_set}"

    best_csv = out_dir / f"y_{top}" / model_name / "best_configs.csv"
    if not os.path.isfile(best_csv):
        raise FileNotFoundError(f"Missing best CSV: {best_csv}. Run best_hyperparams first.")

    # Load hyperparameters from best run
    row = pd.read_csv(best_csv).iloc[0]
    hidden_channels = json.loads(row["hidden_channels"])
    dropout = float(row["dropout"])
    K = int(row["K"]) if ("K" in row.index and not pd.isna(row["K"])) else 3
    window = int(row["window"])
    weighted = bool(row["weighted"])

    # Rewired graph filepath
    graph_fp = os.path.join(data_dir, f"graph_{iteration}_{pct_rewired}.npz")
    if not os.path.isfile(graph_fp):
        raise FileNotFoundError(f"Missing rewired graph file: {graph_fp}")

    # Output dir: rewire (not repeat)
    base_out = out_dir / f"y_{top}" / model_name / "rewire"
    ensure_dir(base_out)

    # Data paths (override graph fp)
    paths = data_paths(data_dir, feature_set, window, top, graph_fp_override=graph_fp)

    # Years / targets
    graph_years = list(range(2010, 2021))
    test_years = [2018, 2019, 2020]
    target_class = 0
    model_kind = MODEL_REGISTRY[model_name]["kind"]

    print(
        f"[REWIRE_SETUP] model={model_name} feature={feature_set} top={top} "
        f"iter={iteration} pct_rewired={pct_rewired} "
        f"weighted={weighted} window={window} graph_fp={graph_fp}"
    )

    # Train + eval
    if model_kind == "static":
        dataset = transform_data(
            model_name=model_name,
            graph_fp=paths["graph_fp"],
            features_fp=paths["features_fp"],
            labels_fp=paths["labels_fp"],
            train_mask_fp=paths["train_mask_fp"],
            val_mask_fp=paths["val_mask_fp"],
            test_mask_fp=paths["test_mask_fp"],
            weighted=weighted,
            target_class=target_class,
            top=top,
        )

        all_y_true, all_y_pred, all_y_probs = [], [], []

        for year in test_years:
            snapshot = get_snapshot_for_hire_year(year, dataset, graph_years)
            num_features = snapshot.x.shape[1]

            model_cls = MODEL_REGISTRY[model_name]["class"]
            model = model_cls(
                num_features=num_features,
                hidden_channels=hidden_channels,
                num_classes=2,
                dropout=dropout,
            )

            trained_model = train_static(snapshot, model, epochs=500, patience=50)

            mcc_y, p_y, r_y, f1_y, cm, pr_auc_y, y_true, y_pred, y_probs = evaluate_static(trained_model, snapshot)
            print(
                f"  [YEAR] {year} | MCC={mcc_y:.3f} | P={p_y:.3f} | R={r_y:.3f} | "
                f"F1={f1_y:.3f} | PR-AUC={pr_auc_y:.3f}"
            )

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_probs.append(y_probs)

        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        all_y_probs = np.concatenate(all_y_probs)

        pr_auc = float(average_precision_score(all_y_true, all_y_probs))
        mcc = float(matthews_corrcoef(all_y_true, all_y_pred))
        accuracy = float(accuracy_score(all_y_true, all_y_pred))
        precision = float(precision_recall_fscore_support(all_y_true, all_y_pred, average="binary")[0])
        recall = float(precision_recall_fscore_support(all_y_true, all_y_pred, average="binary")[1])
        f1 = float(precision_recall_fscore_support(all_y_true, all_y_pred, average="binary")[2])

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_year_indices = [graph_years.index(y) for y in test_years]

        temporal_data = transform_data(
            model_name="GConvGRU",
            graph_fp=paths["graph_fp"],
            features_fp=paths["features_fp"],
            labels_fp=paths["labels_fp"],
            train_mask_fp=paths["train_mask_fp"],
            val_mask_fp=paths["val_mask_fp"],
            test_mask_fp=paths["test_mask_fp"],
            weighted=weighted,
            target_class=0,
            top=top,
        )

        in_channels = temporal_data[0].x.shape[1]
        model_cls = MODEL_REGISTRY[model_name]["class"]
        model = model_cls(
            in_channels=in_channels,
            out_channels=2,
            K=K,
            hidden_channels=hidden_channels,
        ).to(device)

        for snapshot in temporal_data:
            snapshot.x = snapshot.x.to(device)
            snapshot.edge_index = snapshot.edge_index.to(device)
            snapshot.edge_weight = snapshot.edge_weight.to(device)
            snapshot.y = snapshot.y.to(device)
            snapshot.train_mask = snapshot.train_mask.to(device)
            snapshot.val_mask = snapshot.val_mask.to(device)
            snapshot.test_mask = snapshot.test_mask.to(device)

        train_temporal_data = [snap for i, snap in enumerate(temporal_data) if i not in test_year_indices]
        trained_model = train_temporal(train_temporal_data, model, device=device, epochs=500)

        mcc, precision, recall, f1, pr_auc, accuracy = evaluate_temporal(trained_model, temporal_data, test_year_indices)
        pr_auc = float(pr_auc)
        mcc = float(mcc)
        precision = float(precision)
        recall = float(recall)
        f1 = float(f1)
        accuracy = float(accuracy)

        print(
            f"  [TEMP] PR-AUC={pr_auc:.3f} | MCC={mcc:.3f} | P={precision:.3f} | "
            f"R={recall:.3f} | F1={f1:.3f} | Acc={accuracy:.3f}"
        )

    # Save metrics CSV (one per job)
    metrics = dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        mcc=mcc,
        model_name=model_name,
        feature_set=f'{feature_set}+graph',
        top=int(top),
        weighted=bool(weighted),
        target_class=0,
        pct_rewired=float(pct_rewired),
        iteration=int(iteration),
        graph_fp=str(graph_fp),
        window=int(window),
        hidden_channels=json.dumps(hidden_channels),
        dropout=float(dropout),
        K=int(K) if model_kind != "static" else None,
    )

    out_csv = base_out / f"run_iter{iteration}_pct{pct_rewired}.csv"
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)


def analyze_rewire():
    """
    Reads results from all rewired experiments and aggregates into a summary CSV.

    :return: None
    """

    rows = []

    for feat in ['cv', 'bib', 'cv+bib']:
        feat_dir = OUTPUT_DIR / f'models_{feat}' / 'y_10'
        if not feat_dir.exists():
            continue

        for model in ['GCN', 'GAT', 'GraphSAGE', 'GConvGRU']:
            rewire_dir = feat_dir / model / 'rewire'
            if not rewire_dir.exists():
                continue

            # Read all run_*.csv inside that folder
            for fp in sorted(rewire_dir.glob("run_iter*_pct*.csv")):
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    continue

                if df.empty:
                    continue

                # Ensure the identifiers exist even if something got dropped
                df["model_name"] = df.get("model", model)
                df["feature_set"] = df.get("feature", feat)

                # percent is called pct_rewired in your rewire() output
                if "pct_rewired" in df.columns:
                    df["percent"] = df["pct_rewired"]
                elif "percent" not in df.columns:
                    # Try to parse from filename as fallback
                    m = re.search(r"pct([0-9.]+)", fp.name)
                    df["percent"] = m.group(1) if m else None

                # iteration is explicitly in the CSV, but also parseable from filename
                if "iteration" not in df.columns:
                    m = re.search(r"iter([0-9]+)", fp.name)
                    df["iteration"] = int(m.group(1)) if m else None

                rows.append(df)

    if not rows:
        raise SystemExit(
            f"No rewiring CSVs found under {OUTPUT_DIR}/models_*/y_10/*/rewire/.\n"
            "Double-check --output-dir and --top."
        )

    df = pd.concat(rows, ignore_index=True)

    # Save the analysis file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = OUTPUT_DIR / "rewired_summary.csv"
    df.to_csv(out_fp, index=False)


def main():
    """
    Runs graph machine learning experiments:

    1) hyperparam_sweep
        A parallelized hyperparameter sweep that runs one configuration at a time based on command-line input
    2) best_hyperparams
        Analyzes the hyperparameter sweep and identifies the best model + feature set + top combinations
    3) repeat_best
        Retrains the best hyperparameter combination of model + feature set + top combination n_runs times
    4) analyze_repeated_runs
        Creates summary dataframes with results from best repeated runs
    5) rewire
        Runs experiments on rewired graphs
    6) analyze_rewire
        Creates summary dataframe with results from rewired experiments

    :return: None
    """
    if len(sys.argv) < 2:
        print("Usage: python graph_ml.py <experiment> ...")
        sys.exit(1)

    experiment = sys.argv[1]

    # parallelized hyperparameter sweep (one at a time based on command-line input)
    if experiment == 'hyperparam_sweep':

        if len(sys.argv) != 6:
            print("Usage: python graph_ml.py hyperparam_sweep <model_name> <feature_set> <top> <sweep_idx>")
            print("Example: python graph_ml.py hyperparam_sweep GCN biblio 20 5")
            sys.exit(1)

        model_name = sys.argv[2]
        feature_set = sys.argv[3]
        top = int(sys.argv[4])
        sweep_idx = int(sys.argv[5])

        parallelized_hyperparameter_sweep(model_name, feature_set, top, sweep_idx)

    # identify the best hyperparameters for each model+feature+top combination
    elif experiment == 'best_hyperparams':

        if len(sys.argv) != 5:
            print("Usage: python graph_ml.py best_hyperparams <model_name> <feature_set> <top>")
            print("Example: python graph_ml.py best_hyperparams GCN biblio 20")
            sys.exit(1)

        model_name = sys.argv[2]
        feature_set = sys.argv[3]
        top = int(sys.argv[4])

        find_best_hyperparams(model_name, feature_set, top)

    # retrain model n_runs times with the best hyperparams
    elif experiment == 'repeat_best':

        if len(sys.argv) != 6:
            print("Usage: python graph_ml.py repeat_best <model_name> <feature_set> <top> <n_runs>")
            print("Example: python graph_ml.py repeat_best GCN biblio 20 10")
            sys.exit(1)

        model_name = sys.argv[2]
        feature_set = sys.argv[3]
        top = int(sys.argv[4])
        n_runs = int(sys.argv[5])

        repeat_best_runs(model_name, feature_set, top, n_runs)

    # combine dataframes and analyze repeated run performance
    elif experiment == 'analyze_repeated_runs':

        if len(sys.argv) != 3:
            print("Usage: python graph_ml.py analyze_repeated_runs <top>")
            print("Example: python graph_ml.py analyze_repeated_runs 20")
            sys.exit(1)

        top = int(sys.argv[2])

        analyze_repeated_runs_for_top(top)

    # to train on rewired graphs
    elif experiment == "rewire":
        if len(sys.argv) < 7:
            print("Usage: python graph_ml.py rewire_best <model_name> <feature_set> <top> <pct_rewired> <iteration>")
            sys.exit(1)

        model_name = sys.argv[2]
        feature_set = sys.argv[3]
        top = int(sys.argv[4])
        pct_rewired = sys.argv[5]
        iteration = int(sys.argv[6])

        rewire(
            model_name=model_name,
            feature_set=feature_set,
            top=top,
            pct_rewired=pct_rewired,
            iteration=iteration,
        )

    # to analyze rewired experiments
    elif experiment == "analyze_rewire":
        if len(sys.argv) != 2:
            print("Usage: python graph_ml.py analyze_rewire")
            sys.exit(1)

        analyze_rewire()

    else:
        raise NotImplementedError


if __name__=='__main__':
    main()