"""
feature_disentangling.py

Computes VIF and Spearman Correlation for tabular features,
and generates the Spearman Correlation heatmap for the paper.
Also computes feature distibutions for Appendix Table A1.

1/22/2026 - SD
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"


def read_feature_list(path):
    """
    Read a newline-delimited feature list file.

    :param path: path to file
    :return: feats - features
    """
    feats = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    return feats


def ensure_dir(path):
    """
    Guarantees the directory exists.

    :param path: path to directory
    :return: None
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_feature_matrix(data_csv, cv_features_path, bib_features_path):
    """
    Load the faculty dataframe and extract the requested features.

    :param data_csv: faculty csv to read
    :param cv_features_path: file with cv features
    :param bib_features_path: file with bib features
    :return: X - dataframe with features, cv_features - list of feats,
             bib_features - list of feats, all_features - list of feats
    """
    df = pd.read_csv(data_csv)

    cv_features = read_feature_list(cv_features_path)
    bib_features = read_feature_list(bib_features_path)
    all_features = cv_features + bib_features

    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing features in {data_csv}:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    X = df[all_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, cv_features, bib_features, all_features


def compute_spearman_corr(X):
    """
    Compute Spearman correlation matrix.

    :param X: feature matrix
    :return: correlation matrix
    """
    return X.corr(method="spearman")


def cluster_order_from_corr(corr):
    """
    Compute a hierarchical clustering order for the correlation matrix.

    :param corr: correlation matrix
    :return: clustering order
    """
    C = corr.to_numpy(dtype=float)
    C = np.nan_to_num(C, nan=0.0)
    dist = 1.0 - np.abs(C)
    np.fill_diagonal(dist, 0.0)

    # squareform expects condensed distance vector
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return order.tolist()


def plot_corr_heatmap(corr, out_path, title, cluster=True, figsize=(12, 10), vmin=-1.0, vmax=1.0):
    """
    Plot and save a correlation heatmap. If cluster=True and SciPy is available, reorder rows/cols via
    hierarchical clustering.

    :param corr: Correlation matrix
    :param out_path: output file path
    :param title: title for heatmap
    :param cluster: whether to cluster
    :param figsize: fig size
    :param vmin: minimum cmap value
    :param vmax: maximum cmap value
    :return:
    """
    corr_plot = corr.copy()

    pretty_names = {'doctoral_ranking': 'PhD University Rank', 'gender': 'Gender', 'pre_academic_age': 'Academic Age',
                    'cum_num_log_papers': 'log(# Papers)', 'cum_avg_authors_per_paper': 'Avg. Authors per Paper',
                    'cum_num_log_first_authored': 'log(# First-authored)', 'cum_avg_author_position': 'Avg. Author Position',
                    'cum_num_log_high_faculty_papers': 'log(# with High-rank Fac.)', 'cum_num_log_med_faculty_papers': 'log(# with Med.-rank Fac.)',
                    'cum_num_log_citations': 'log(# Citations)'}

    if cluster:
        order = cluster_order_from_corr(corr_plot)
        corr_plot = corr_plot.iloc[order, order]

    corr_plot = corr_plot.rename(index=pretty_names, columns=pretty_names)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_plot.values, vmin=vmin, vmax=vmax, aspect="auto", cmap='BrBG')

    ax.text(
        -0.3, 1.1,
        title,
        fontweight='bold',
        ha='left',
        va='top',
        color='#333333',
        fontsize=9,
        transform=ax.transAxes
    )

    ax.set_xticks(range(len(corr_plot.columns)))
    ax.set_yticks(range(len(corr_plot.index)))
    ax.set_xticklabels(corr_plot.columns, rotation=30, fontsize=7, ha="right")
    ax.set_yticklabels(corr_plot.index, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r"Spearman Correlation $\rho$", rotation=270, labelpad=10, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def correlations(X, all_feats):
    """
    Generates correlation matrix and plots heatmap.

    :param X: feature matrix
    :param all_feats: features to include
    :return: None
    """
    # Compute Spearman correlation
    corr = compute_spearman_corr(X)

    # Save correlation matrix
    corr_csv = os.path.join(PLOTS_DIR, "spearman_corr_all_features.csv")
    corr.to_csv(corr_csv)

    # Heatmap
    heatmap_path = os.path.join(PLOTS_DIR, "spearman_corr_heatmap_all.pdf")
    plot_corr_heatmap(
        corr=corr.loc[all_feats, all_feats],
        out_path=heatmap_path,
        title=f"Spearman Correlation between Features",
        cluster=True,
        figsize=(7.2, 3.2),
    )


def standardize(df):
    """
    Standardize features.

    :param df: dataframe with features
    :return: standardized dataframe
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - means) / stds


def prepare_matrix_for_vif(df, features):
    """
    Standardize features for VIF calculation.

    :param df: dataframe with features
    :param features: list of features to include
    :return: standardized feature matrix and list of columns
    """
    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    X = standardize(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="any")

    kept = list(X.columns)

    if X.shape[1] > 0:
        X = sm.add_constant(X, has_constant="add")

    return X, kept


def compute_vif_table(df, features):
    """
    Compute VIF for a set of features.

    :param df: dataframe with features
    :param features: list of features
    :return: dataframe with VIFs
    """

    X, kept = prepare_matrix_for_vif(df, features)

    # If only 0/1 features remain, VIF isn't meaningful
    if X.shape[1] <= 2 and "const" in X.columns:
        rows = [{"feature": f, "vif": np.nan, "flagged": False} for f in kept]
        return pd.DataFrame(rows, columns=["feature", "vif", "flagged"])

    rows = []
    cols = list(X.columns)
    for i, col in enumerate(cols):
        if col == "const":
            continue
        vif_val = float(variance_inflation_factor(X.values, i))
        rows.append({"feature": col, "vif": vif_val})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)
    return out


def multicollinearity(all_feats, cv_feats, bib_feats):
    """
    Calculate multicollinearity for various feature sets.

    :param all_feats: CV + Bib features
    :param cv_feats: CV features
    :param bib_feats: Bib features
    :return: None
    """
    faculty_df = pd.read_csv('../data/hiring/faculty.csv')

    vif_cv = compute_vif_table(faculty_df, cv_feats)
    vif_bib = compute_vif_table(faculty_df, bib_feats)
    vif_all = compute_vif_table(faculty_df, all_feats)

    out_cv = os.path.join(PLOTS_DIR, f"vif_cv.csv")
    out_bib = os.path.join(PLOTS_DIR, f"vif_bib.csv")
    out_all = os.path.join(PLOTS_DIR, f"vif_cv_plus_bib.csv")

    vif_cv.to_csv(out_cv, index=False)
    vif_bib.to_csv(out_bib, index=False)
    vif_all.to_csv(out_all, index=False)


def finite(x):
    """
    Return finite values.

    :param x: feature series
    :return: finite feature array
    """
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def get_feature_vector(df, feature, transform="none", clip_nonneg=True,):
    """
    Extract a feature vector aligned to df row order (node id == row index),
    optionally applying a transform.

    :param df: dataframe
    :param feature: feature
    :param transform: transformation (none or log)
    :param clip_nonneg: whether to make non-negative
    :return: feature vector
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in dataframe columns.")

    x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)

    if clip_nonneg:
        # for count-like features; harmless for others if already >=0
        x = np.where(np.isfinite(x), x, np.nan)
        x = np.maximum(x, 0.0)

    if transform == "none":
        pass
    elif transform == "log1p":
        x = np.log1p(x)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    return x


def feature_dists(x_raw):
    """
    Compute distribution / tail stats on the (transformed, non-negative) raw features.

    :param x_raw: raw features
    :return:
    """
    xf = finite(x_raw)
    s = pd.Series(xf)
    frac_zero = float(np.mean(xf == 0.0))

    return {
        "n_finite": int(xf.size),
        "frac_zero": frac_zero,
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "mean": float(np.mean(xf)),
        "std": float(np.std(xf)),
        "median": float(np.median(xf)),
        "p90": float(np.percentile(xf, 90)),
        "p99":  float(np.percentile(xf, 99)),
        "p999":  float(np.percentile(xf, 99.9)),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt())
    }


def feature_distribution_stats(cv_feats, bib_feats):
    """
    Calculate feature distributions including skew and kurtosis for raw and log-transformed features.

    :param cv_feats: CV features
    :param bib_feats: Bib features
    :return: None
    """

    faculty_df = pd.read_csv('../data/hiring/faculty.csv')

    base_features = list(dict.fromkeys(cv_feats + bib_feats))

    # Build evaluation list with optional log transforms
    eval_specs = []
    for f in base_features:
        eval_specs.append((f, "none"))
        if ("num_" in f) or ("count" in f) or ("cit" in f) or f.startswith("cum_") or f.startswith("pre_"):
            eval_specs.append((f, "log1p"))

    # Remove duplicates while preserving order
    seen = set()
    eval_specs_unique = []
    for f, t in eval_specs:
        key = (f, t)
        if key in seen:
            continue
        seen.add(key)
        eval_specs_unique.append((f, t))

    dist_rows = []

    for feat, transform in eval_specs_unique:
        x_raw = get_feature_vector(faculty_df, feat, transform=transform, clip_nonneg=True)
        x_raw = np.where(np.isfinite(x_raw), x_raw, 0.0)

        dist = feature_dists(x_raw)
        dist_rows.append({
            "feature_base": feat,
            "transform": transform,
            "feature": f"{feat}" if transform == "none" else f"{transform}({feat})",
            **dist,
        })

    df_dist = pd.DataFrame(dist_rows)
    out_dist_csv = os.path.join(PLOTS_DIR, "feature_distribution_metrics.csv")
    df_dist.to_csv(out_dist_csv, index=False)


def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    # Load
    X, cv_feats, bib_feats, all_feats = load_feature_matrix(
        data_csv='../data/hiring/faculty.csv',
        cv_features_path='../data/hiring/cv_features.txt',
        bib_features_path='../data/hiring/bibliometric_features.txt.txt',
    )

    correlations(X, all_feats)
    multicollinearity(all_feats, cv_feats, bib_feats)
    feature_distribution_stats(cv_feats, bib_feats)