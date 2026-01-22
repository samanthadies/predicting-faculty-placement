"""
tabular_ml.py

Tabular modeling pipeline for faculty placement prediction. This script
1) Prepares features from configurable sets: CV features, bibliometric features, or both.
2) Grid-search style enumeration of classic ML models (LR/RF/GB/SVC/MLP) and a
   TabTransformer; train/evaluate one-vs-all classifiers (high-vs-all).
3) Saves the best per-class models and a CSV summary of results.
4) Optionally re-trains the best found configs multiple times to measure stability.

I/O:
- Inputs: '../data/hiring/faculty.csv' plus feature-name lists in '../data/hiring/*.txt'
- Outputs:
    ../output/best_model__{model}__{feature_set}__{class}.pkl
    ../output/experiment_summary.csv
    ../output/repeat_results.csv

1/22/2026 - SD
"""

import pandas as pd
import numpy as np
import warnings
import copy
import joblib
import torch
import os
from pathlib import Path
from ast import literal_eval
from itertools import product

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from models.TabTransformer import TabTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, classification_report

import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


# Feature sets (already defined)
feature_sets = [
    {"name": "cv", "use_cv": True, "use_biblio": False},
    {"name": "biblio", "use_cv": False, "use_biblio": True},
    {"name": "cv+biblio", "use_cv": True, "use_biblio": True},
]


# Model definitions + param grids
model_registry = {
    "logreg": {
        "model": LogisticRegression,
        "param_grid": {
            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            "class_weight": [None, "balanced"],
            "solver": ["liblinear", "saga"],
            "penalty": ["l1", "l2"],
            "max_iter": [1000],
        },
        "scale": [True, False],
        "sampling": [None, "oversample", "undersample"],
        "allowed_features": {"cv", "biblio", "cv+biblio"},
    },
    "rf": {
        "model": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "class_weight": [None, "balanced"],
        },
        "scale": [False],
        "sampling": [None, "oversample", "undersample"],
        "allowed_features": {"cv", "biblio", "cv+biblio"},
    },
    "xgb": {
        "model": GradientBoostingClassifier,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_split": [2, 5],
        },
        "scale": [False],
        "sampling": [None, "oversample", "undersample"],
        "allowed_features": {"cv", "biblio", "cv+biblio"},
    },
    "svc": {
        "model": SVC,
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10, 100],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
            "probability": [True],
        },
        "scale": [True],
        "sampling": [None, "oversample"],
        "allowed_features": {"cv", "biblio", "cv+biblio"},
    },
    "mlp": {
        "model": MLPClassifier,
        "param_grid": {
            "hidden_layer_sizes": [(100,), (100, 50), (128, 64), (128, 64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
            "activation": ["relu", "tanh"],
            "learning_rate_init": [0.001, 0.01],
            "max_iter": [300],
        },
        "scale": [True],
        "sampling": [None, "oversample"],
        "allowed_features": {"biblio", "cv+biblio"},
    },
    "tab_transformer": {
        "model": TabTransformer,
        "param_grid": {
            "emb_dim": [16, 32, 64],
            "num_heads": [2, 4, 8],
            "num_layers": [1, 2, 3],
            "mlp_hidden": [[100], [100, 50], [128, 64], [128, 64, 32]],
            "dropout": [0.0, 0.1, 0.2],
            "num_classes": [2],
        },
        "scale": [True],
        "sampling": [None],
        "allowed_features": {"biblio", "cv+biblio"},
    }
}


def ensure_dir(p):
    """
    Ensures that files exist.

    :param p: filepath to check
    :return: None
    """
    Path(p).mkdir(parents=True, exist_ok=True)


def prepare_tabtransformer_data(df, categorical_cols, continuous_cols, label_col='y', scalers=None, encoders=None, fit=True):
    """
    Prepare tensors for a TabTransformer-like model.

    :param df: Input DataFrame containing features and label (pd.DataFrame)
    :param categorical_cols: List of column names to treat as categorical (List[str])
    :param continuous_cols: List of column names to treat as continuous (List[str])
    :param label_col: Name of the label column (str)
    :param scalers: Optional dict to pass in a pre-fitted scaler
    :param encoders: Optional dict of pre-fitted LabelEncoders keyed by column name
    :param fit: If True, fit scaler/encoders on `df`; if False, use provided ones
    :return: Tuple (x_cont, x_cat, y, scalers, encoders)
        - x_cont: torch.FloatTensor of shape [N, d_cont]
        - x_cat:  torch.LongTensor  of shape [N, d_cat]
        - y:      torch.LongTensor  of shape [N]
        - scalers: dict with key 'scaler' (StandardScaler)
        - encoders: dict of LabelEncoders per categorical column
    """
    df = df.copy()

    # Label encode categorical columns
    encoders = encoders or {}
    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    x_cat = torch.tensor(df[categorical_cols].values, dtype=torch.long)

    # Standard scale continuous columns
    scalers = scalers or {}
    if continuous_cols:
        if fit:
            scaler = StandardScaler()
            df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
            scalers['scaler'] = scaler
        else:
            scaler = scalers['scaler']
            df[continuous_cols] = scaler.transform(df[continuous_cols])
        x_cont = torch.tensor(df[continuous_cols].values, dtype=torch.float32)
    else:
        x_cont = torch.empty(len(df), 0)

    y = torch.tensor(df[label_col].values, dtype=torch.long)

    return x_cont, x_cat, y, scalers, encoders


def format_train_test(train_path='../data/hiring/faculty_train.csv', val_path='../data/hiring/faculty_val.csv',
                      test_path='../data/hiring/faculty_test.csv', cv_features_file='../data/hiring/cv_features.txt',
                      biblio_features_file='../data/hiring/bibliometric_features.txt', use_cv=False, use_biblio=False,
                      scale=False, sampling=None, top=10):
    """
    Build X/y matrices for train/val/test using selected features.

    This function also supports:
      - one-hot encoding of 'sub_field' when selected in feature lists
      - standardization of numeric features (based on train only)
      - optional train-set resampling ('oversample'/'undersample')

    :param train_path: Path to the training split CSV
    :param val_path: Path to the validation split CSV
    :param test_path: Path to the testing split CSV
    :param cv_features_file: Text file listing CV feature names (one per line)
    :param biblio_features_file: Text file listing bibliometric features (one per line)
    :param use_cv: If True, include CV features
    :param use_biblio: If True, include bibliometric features
    :param scale: If True, standardize numeric features with StandardScaler
    :param sampling: One of {None, 'oversample', 'undersample'} applied to the training set
    :param top: What y's to consider as High-rank
    :return: Dict with keys:
        - 'X_train', 'X_val', 'X_test' (pd.DataFrame)
        - 'y_train', 'y_val', 'y_test' (np.ndarray, categorical)
        - 'feature_names' (List[str])
        - 'y_train_bin'/'y_val_bin'/'y_test_bin' (dicts of np.ndarray with class name 'high')
        - If sampling: 'X_train_<sampling>', 'y_train_<sampling>' (and bin variants)
    """

    # load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # convert 'y' to categorical for classification
    for df in [train_df, val_df, test_df]:
        if top == 10:
            ybin = (df['y'].astype(int) == 0).astype(int)
        else:
            ybin = df[f'y_{top}'].astype(int)

        df['y_bin'] = ybin

    # load feature lists from files
    selected_features = []
    if use_cv and cv_features_file:
        with open(cv_features_file, 'r') as f:
            selected_features += [line.strip() for line in f.readlines()]
    if use_biblio and biblio_features_file:
        with open(biblio_features_file, 'r') as f:
            selected_features += [line.strip() for line in f.readlines()]

    # one-hot encode sub_field if it's selected
    if 'sub_field' in selected_features:
        all_subfields = pd.get_dummies(
            pd.concat([train_df['sub_field'], val_df['sub_field'], test_df['sub_field']]),
            prefix='subfield'
        ).columns

        def encode_subfields(df):
            dummies = pd.get_dummies(df['sub_field'], prefix='subfield')
            for col in all_subfields:
                if col not in dummies:
                    dummies[col] = 0
            return dummies[all_subfields]
    else:
        def encode_subfields(df):
            return pd.DataFrame(index=df.index)

    # determine numeric features to scale (skip sub_field)
    numeric_features = [f for f in selected_features if f != 'sub_field']

    # fit scaler on numeric features (excluding categorical dummies)
    scaler = StandardScaler()
    if scale and numeric_features:
        scaler.fit(train_df[numeric_features])

    def build_X(df):
        # numeric features
        X = df[numeric_features].copy() if numeric_features else pd.DataFrame(index=df.index)
        if scale and numeric_features:
            X[numeric_features] = scaler.transform(X[numeric_features])
        # add subfield dummies if included
        if 'sub_field' in selected_features:
            X = pd.concat([X.reset_index(drop=True), encode_subfields(df).reset_index(drop=True)], axis=1)
        return X

    # build train and test data
    X_train = build_X(train_df)
    y_train = train_df['y_bin'].astype(int).to_numpy()

    X_val = build_X(val_df)
    y_val = val_df['y_bin'].astype(int).to_numpy()

    X_test = build_X(test_df)
    y_test = test_df['y_bin'].astype(int).to_numpy()

    # create data dict (formatted for binary labels)
    data = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'feature_names': X_train.columns.tolist(),
        'y_train_bin': {'high': y_train},
        'y_val_bin': {'high': y_val},
        'y_test_bin': {'high': y_test},
    }

    # optional: Sampling
    if sampling in ['oversample', 'undersample']:
        sampler = RandomOverSampler(random_state=42) if sampling == 'oversample' else RandomUnderSampler(
            random_state=42)

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        data[f'X_train_{sampling}'] = X_resampled
        data[f'y_train_{sampling}'] = y_resampled

        X_resampled, y_resampled = sampler.fit_resample(X_train, data['y_train_bin']['high'])
        data[f'X_train_bin_{sampling}_high'] = X_resampled
        data[f'y_train_bin_{sampling}_high'] = y_resampled

    elif sampling is not None:
        raise NotImplementedError

    return data


def train_tabtransformer( model, train_data, train_labels, val_data, val_labels, max_epochs=50, lr=1e-3, patience=5,
                          verbose=True):
    """
    Train a TabTransformer in a minimal loop with early stopping on AUROC.

    :param model: TabTransformer instance
    :param train_data: Float tensor of shape [N_train, d] with continuous features
    :param train_labels: Long tensor of shape [N_train] with {0,1} labels
    :param val_data: Float tensor of shape [N_val, d] for validation
    :param val_labels: Long tensor of shape [N_val] with {0,1} labels
    :param max_epochs: Maximum epochs to train
    :param lr: Learning rate for Adam optimizer
    :param patience: Early stopping patience (epochs without AUROC improvement)
    :param verbose: If True, print training progress
    :return: Tuple (best_model, best_auc)
        - best_model: model with weights from the best AUROC epoch.
        - best_auc: float AUROC on validation at the best epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_model = None
    best_auc = -1
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_data)
            probs = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            y_true = val_labels.cpu().numpy()
            auc = roc_auc_score(y_true, probs)

        if verbose:
            print(f"Epoch {epoch+1}/{max_epochs} - Train loss: {loss.item():.4f} - Val AUROC: {auc:.4f}")

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    # Restore best weights
    model.load_state_dict(best_model)

    return model, best_auc


def train_model(model, X_train, y_train):
    """
    Fit an sklearn-style estimator.

    :param model: Estimator with .fit(X, y)
    :param X_train: Training feature matrix (pd.DataFrame or np.ndarray)
    :param y_train: Training labels (array-like)
    :return: The fitted model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y_true, label='', verbose=False):
    """
    Evaluate a fitted model and print a classification report.

    :param model: Fitted estimator with .predict(X)
    :param X: Feature matrix to evaluate on
    :param y_true: Ground-truth labels (array-like)
    :param label: Optional label for the printed section header (e.g., 'val' or 'test')
    :param verbose: If True, print results
    :return: Tuple (y_pred, acc)
        - y_pred: np.ndarray of predicted class labels
        - acc: float accuracy
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"\n=== {label.upper()} SET ===")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_true, y_pred))

    return y_pred, acc


def train_and_evaluate(
    model,
    data,
    label_key='y_train',
    eval_keys=('y_val', 'y_test'),
    X_key='X_train',
    verbose=True
):
    """
    Fit model on the provided training split and evaluate on specified splits.

    :param model: Estimator with .fit/.predict
    :param data: Dict produced by format_train_test() containing X_* and y_*
    :param label_key: Key in 'data' for training labels (e.g., 'y_train' or 'y_train_bin'[...])
    :param eval_keys: Iterable of label keys to evaluate on (e.g., ('y_val','y_test'))
    :param X_key: Key in `data` for training features (default: 'X_train')
    :param verbose: If True, print metrics
    :return: Dict with:
        - 'model': fitted model
        - '{ek}_pred' and '{ek}_acc' for each ek in eval_keys
    """
    y_train = data[label_key]
    X_train = data[X_key]
    model = train_model(model, X_train, y_train)

    results = {'model': model}

    for eval_key in eval_keys:
        y_eval = data[eval_key]
        X_eval = data[eval_key.replace('y_', 'X_')]
        preds, acc = evaluate_model(model, X_eval, y_eval, label=eval_key.replace('y_', ''), verbose=verbose)
        results[f'{eval_key}_pred'] = preds
        results[f'{eval_key}_acc'] = acc

    # Optionally log coefficients
    if verbose:
        if hasattr(model, 'coef_'):
            print("\n=== COEFFICIENTS ===")
            for i, class_label in enumerate(model.classes_):
                print(f"\nClass {class_label}:")
                for name, coef in zip(data['feature_names'], model.coef_[i]):
                    print(f"  {name:<35} {coef: .4f}")

        if hasattr(model, 'feature_importances_'):
            print("\n=== FEATURE IMPORTANCES ===")
            for name, imp in zip(data['feature_names'], model.feature_importances_):
                print(f"  {name:<35} {imp: .4f}")

    return results


def generate_model_configs(model_name, model_info, feature_config):
    """
    Enumerate all hyperparameter combinations x (scale, sampling) for a model.

    :param model_name: Registry key for the model (e.g., 'logreg', 'rf', 'svc')
    :param model_info: Dict from `model_registry` for this model (includes 'model', 'param_grid', 'scale', 'sampling')
    :param feature_config: One entry from `feature_sets` (e.g., {'name': 'cv+biblio', 'use_cv': True, ...})
    :return: List of config dicts, each with keys:
        - 'model_name', 'model_class', 'model_params', 'feature_config', 'scale', 'sampling'
    """
    param_grid = model_info['param_grid']
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # All hyperparam combinations
    all_param_combos = list(product(*param_values))

    configs = []
    for param_combo in all_param_combos:
        param_dict = dict(zip(param_names, param_combo))

        for scale in model_info['scale']:
            for sampling in model_info['sampling']:
                config = {
                    'model_name': model_name,
                    'model_class': model_info['model'],
                    'model_params': copy.deepcopy(param_dict),
                    'feature_config': feature_config,
                    'scale': scale,
                    'sampling': sampling,
                }
                configs.append(config)
    return configs


def run_all_experiments(config_list, save_dir='../output', top=10):
    """
    Run all configs; for each (model, feature_set), keep best avg F1 on test.
    Saves each best per-class model to disk and writes a CSV summary.

    :param config_list: Sequence of config dicts produced by generate_model_configs()
    :param save_dir: Directory to save models and summary CSV
    :param top: What y's to consider as High-rank
    :return: pd.DataFrame summary with one row per saved (model, feature_set, class) including params and score
    """
    os.makedirs(save_dir, exist_ok=True)
    summary = []
    best_config_results = {}

    # preload dataframes once
    train_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_train.csv')
    val_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_val.csv')
    test_df = pd.read_csv(DATA_DIR / 'hiring' / 'faculty_test.csv')

    for config in config_list:
        # dynamically determine selected features based on config
        cv_features_file = DATA_DIR / 'hiring' / 'cv_features.txt'
        biblio_features_file = DATA_DIR / 'hiring' / 'bibliometric_features.txt'

        selected_features = []
        if config['feature_config']['use_cv']:
            with open(cv_features_file, 'r') as f:
                selected_features += [line.strip() for line in f.readlines()]
        if config['feature_config']['use_biblio']:
            with open(biblio_features_file, 'r') as f:
                selected_features += [line.strip() for line in f.readlines()]

        # separate continuous and categorical features
        continuous_cols = [col for col in selected_features if col in train_df.columns and col != 'sub_field']

        model_name = config['model_name']
        feature_name = config['feature_config']['name']
        key = (model_name, feature_name)

        per_class_f1 = {}
        per_class_model = {}

        print(f"Running: {key} | High-vs-all")

        # train TabTransformer model
        if model_name == 'tab_transformer':
            if top == 10:
                y_train_bin = (train_df['y'] == 0).astype(int)
                y_val_bin = (val_df['y'] == 0).astype(int)
                y_test_bin = (test_df['y'] == 0).astype(int)
            else:
                y_train_bin = train_df[f'y_{top}']
                y_val_bin = val_df[f'y_{top}']
                y_test_bin = test_df[f'y_{top}']

            X_train = train_df[continuous_cols].values.astype(np.float32)
            X_val = val_df[continuous_cols].values.astype(np.float32)
            X_test = test_df[continuous_cols].values.astype(np.float32)

            model = config['model_class'](
                num_features=X_train.shape[1],
                **config['model_params']
            )

            model, _ = train_tabtransformer(
                model, torch.tensor(X_train), torch.tensor(y_train_bin.values),
                torch.tensor(X_val), torch.tensor(y_val_bin.values),
                max_epochs=100, patience=5
            )

            probs = model.predict_proba(torch.tensor(X_test))
            preds = np.argmax(probs, axis=1)
            f1 = f1_score(y_test_bin, preds)
            per_class_f1['high'] = f1
            per_class_model['high'] = model

        # train other models
        else:
            data = format_train_test(
                use_cv=config['feature_config']['use_cv'],
                use_biblio=config['feature_config']['use_biblio'],
                scale=config['scale'],
                sampling=config['sampling'],
                top=top
            )

            X_train = data[f'X_train_bin_{config["sampling"]}_high'] if config['sampling'] else data['X_train']
            y_train = data[f'y_train_bin_{config["sampling"]}_high'] if config['sampling'] else data['y_train_bin']['high']
            X_test = data['X_test']
            y_test = data['y_test_bin']['high']

            model = config['model_class'](**config['model_params'])
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            f1 = f1_score(y_test, preds)
            per_class_f1['high'] = f1
            per_class_model['high'] = model

        avg_f1 = np.mean(list(per_class_f1.values()))

        if key not in best_config_results or avg_f1 > best_config_results[key]['avg_test_f1']:
            best_config_results[key] = {
                'config': config,
                'avg_test_f1': avg_f1,
                'per_class_f1': per_class_f1,
                'models': per_class_model
            }

    # save results
    for key, result in best_config_results.items():
        model_name, feature_name = key
        config = result['config']
        per_class_f1 = result['per_class_f1']
        models = result['models']

        model = models['high']
        f1 = per_class_f1['high']

        save_dir = Path(save_dir)
        model_path = save_dir / f'best_model__{model_name}__{feature_name}__high.pkl'
        joblib.dump(model, model_path)

        summary.append({
            'model_name': model_name,
            'feature_set': feature_name,
            'target_class': 'high',
            'params': config['model_params'],
            'scale': config['scale'],
            'sampling': config['sampling'],
            'f1': f1,
            'model_path': model_path,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(save_dir / 'experiment_summary.csv', index=False)
    print(f"Finished! Saved {len(summary_df)} best models. Summary written to 'experiment_summary.csv'.")
    return summary_df


def run_cv_biblio_experiments(top=10):
    """
    Generate all (model, feature_set) configurations allowed by the registry and run them (i.e., hyperparameter search).

    :param top: What y's to consider as High-rank
    :return: None
    """
    all_configs = []
    for feature_cfg in feature_sets:
        for model_name, model_info in model_registry.items():
            if feature_cfg["name"] in model_info["allowed_features"]:
                all_configs += generate_model_configs(model_name, model_info, feature_cfg)

    _ = run_all_experiments(all_configs, save_dir=OUTPUT_DIR / f'y_{top}', top=top)


def repeat_results(summary_csv=None, n_runs=10,
                   save_path=None, model_save_dir=None, top=10):
    """
    Re-train/evaluate best per-class models multiple times to estimate variance.

    :param summary_csv: Path to CSV produced by run_all_experiments()
    :param n_runs: Number of re-training runs per (model, feature_set, class)
    :param save_path: Output CSV path for the repeated metrics
    :param model_save_dir: Directory to save the best model per class (overwritten per row)
    :param top: What y's to consider as High-rank
    :return: None
    """
    if summary_csv is None:
        summary_csv = OUTPUT_DIR / 'experiment_summary.csv'
    if save_path is None:
        save_path = OUTPUT_DIR / 'repeat_results.csv'
    if model_save_dir is None:
        model_save_dir = OUTPUT_DIR / 'best_simple_models'

    ensure_dir(model_save_dir)

    df = pd.read_csv(summary_csv)
    results = []

    for _, row in df.iterrows():
        model_name = row['model_name']
        feature_set = row['feature_set']
        cls_label = row['target_class']
        params = literal_eval(row['params'])
        scale = row['scale']
        sampling = row['sampling'] if not pd.isnull(row['sampling']) else None

        best_f1 = -1
        best_model = None

        key = (model_name, feature_set)
        print(f"Running: {key} | class: {cls_label}")

        for run in range(n_runs):
            data = format_train_test(
                use_cv=(feature_set in ['cv', 'cv+biblio']),
                use_biblio=(feature_set in ['biblio', 'cv+biblio']),
                scale=scale,
                sampling=sampling,
                top=top
            )

            X_train = data[f'X_train_bin_{sampling}_{cls_label}'] if sampling else data['X_train']
            y_train = data[f'y_train_bin_{sampling}_{cls_label}'] if sampling else data['y_train_bin'][cls_label]
            X_val = data['X_val']
            y_val = data['y_val_bin'][cls_label]
            X_test = data['X_test']
            y_test = data['y_test_bin'][cls_label]

            # Retrain tab_transformer
            if model_name == 'tab_transformer':
                model = TabTransformer(num_features=X_train.shape[1], **params)
                model, _ = train_tabtransformer(
                    model, torch.tensor(X_train.values.astype(np.float32)), torch.tensor(y_train),
                    torch.tensor(X_val.values.astype(np.float32)), torch.tensor(y_val),
                    max_epochs=100, patience=5
                )
                y_pred = np.argmax(model.predict_proba(torch.tensor(X_test.values.astype(np.float32))), axis=1)
                y_probs = model.predict_proba(torch.tensor(X_test.values.astype(np.float32)))[:, 1]

            # retrain other models
            else:
                model_class = model_registry[model_name]['model']
                model = model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_probs = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model

            results.append({
                'model_name': model_name,
                'feature_set': feature_set,
                'target_class': cls_label,
                'run': run,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_probs),
                'pr_auc': average_precision_score(y_test, y_probs),
                'mcc': matthews_corrcoef(y_test, y_pred)
            })

            # Save best model
            model_path = Path(model_save_dir) / f'retrained_best_model__{model_name}__{feature_set}__{cls_label}.pkl'
            joblib.dump(best_model, model_path)

    pd.DataFrame(results).to_csv(save_path, index=False)


def main():
    """
    Train tabular machine learning models. Includes hyperparameter search and final, repeated runs.

    :return: None
    """
    warnings.filterwarnings('ignore')

    top = [10, 20, 30, 40, 50]
    for t in top:
        # run initial experiments with cv + bibliographic features
        run_cv_biblio_experiments(top=t)

        # repeat experiments with best models
        repeat_results(summary_csv=OUTPUT_DIR / f'y_{t}' / 'experiment_summary.csv',
                       n_runs=10,
                       save_path=OUTPUT_DIR / f'y_{t}' / 'repeat_results.csv',
                       model_save_dir=OUTPUT_DIR / f'y_{t}' / 'best_simple_models', top=t)


if __name__=='__main__':
    main()
