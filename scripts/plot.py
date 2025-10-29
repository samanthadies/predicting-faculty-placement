"""
plot.py

This script generates the plots present in the paper.

10/29/2025 - SD
"""

import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

import explicit_baselines


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
REWIRE_DIR = OUTPUT_DIR / "rewire"
PLOTS_DIR = OUTPUT_DIR / "plots"


plt.rcParams.update({
    "figure.figsize": (7.2, 4.0),
    "savefig.dpi": 600,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.title_fontsize": 8,
    "axes.titlesize": 10,
    "text.usetex": False,
    "font.family": "serif"
})


model_color_map = {
    # White-box models (reds)
    'logreg': '#c96b63',     # lighter
    'svc': '#6b2c20',        # darker, redder

    # Tree-based models (yellows)
    'rf': '#cda633',         # more saturated, less brown
    'xgb': '#7f730d',        # darker, slightly greener

    # Neural networks (blues/teals)
    'mlp': '#3a8b92',        # brighter teal
    'numerical_transformer': '#10373b',  # deeper blue-green

    # GNNs (purples)
    'GCN': '#d69dc2',      # light pinkish-mauve
    'GAT': '#b05fa4',      # medium magenta
    'GraphSAGE': '#7a3f84',# deep violet
    'GConvGRU': '#4b2972'  # dark indigo-purple
}

feature_colors = [
        "#7b8f4e",  # CV: olive green
        "#ce796b",  # Biblio: clay red
        "#d5a021",  # Graph: warm amber
        "#5a9e8c",  # CV+Biblio: slate teal
        "#a36f9a",  # CV+Graph: dusty plum
        "#7c6ab2",  # Biblio+Graph: royal lavender
        "#a8a29e"  # CV+Biblio+Graph: earthy gray
    ]

ordered_models = [
    'logreg', 'svc', 'rf', 'xgb',
    'mlp', 'numerical_transformer',
    'GCN', 'GAT', 'GraphSAGE', 'GConvGRU'
]

ordered_feature_sets = [
        'cv', 'biblio',
        'cv+biblio', 'graph', 'cv+graph', 'biblio+graph', 'cv+biblio+graph'
    ]

pretty_model_labels = {
    'logreg': 'LR',
    'rf': 'RF',
    'svc': 'SVM',
    'xgb': 'XGB',
    'mlp': 'MLP',
    'numerical_transformer': 'Trans.',
    'GCN': 'GCN',
    'GAT': 'GAT',
    'GraphSAGE': 'GraphSAGE',
    'GConvGRU': 'GConvGRU'
}

pretty_feature_labels = {
        'cv': 'PhD Rank',
        'biblio': 'Bibliometric',
        'graph': 'Co-authorship',
        'cv+biblio': 'PhD\n+\nBibliometric',
        'cv+graph': 'PhD Rank\n+\nCo-authorship',
        'biblio+graph': 'Bibliometric\n+\nCo-authorship',
        'cv+biblio+graph': 'PhD Rank\n+\nBibliometric\n+\nCo-authorship'
    }


def ensure_dir(p):
    """
    Ensures that files exist.

    :param p: filepath to check
    :return: None
    """
    Path(p).mkdir(parents=True, exist_ok=True)


def create_figure(figsize, nrows, ncols, height_ratios=None, width_ratios=None, hspace=None, wspace=None):
    """
    Create new figure with gridspec.

    :param figsize: tuple describing the figure size in inches
    :param nrows: number of rows in the gridspec
    :param ncols: number of columns in the gridspec
    :param height_ratios: list of relative heights of gridspec panels
    :param width_ratios: list of relative widths of gridspec panels
    :param hspace: gridspaceing horizontally
    :param wspace: gridspaceing vertically
    :return: the figure and gridspec
    """
    # create the figure
    fig = plt.figure(figsize=figsize)

    # create the gridspec
    gs = grid_spec.GridSpec(
        figure=fig,
        nrows=nrows,
        ncols=ncols,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
        hspace=hspace,
        wspace=wspace
    )

    return fig, gs


def add_text(fig, gs, text, pos, style):
    """
    Add a new axis in the gridspec with text.

    :param fig: the figure
    :param gs: the gridspec panel
    :param text: the text
    :param pos: the position
    :param style: the style
    :return: the figure
    """

    # add new axis
    ax = fig.add_subplot(gs)

    # turn of ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # turn of spines
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    # remove background
    ax.patch.set_alpha(0.)

    # set the title
    x, y = pos
    ax.text(x, y, text, transform=ax.transAxes, fontsize=style['fontsize'], fontweight=style['fontweight'],
            color=style['fontcolor'])

    return fig


def ci_bounds(data, ci=90, n_boot=1000):
    """
    Bootstrap confidence interval for the mean.

    :param data: data to sample
    :param ci: confidence interval width
    :param n_boot: number of samples
    :return: upper and lower bounds
    """
    np.random.seed(0)
    means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


def ci_bounds_differential(base, alt, ci=90, n_boot=1000):
    """
    Bootstrap confidence interval difference between two runs.

    :param base: data from first run
    :param alt: data from second run
    :param ci: confidence interval width
    :param n_boot: number of samples
    :return: upper and lower bounds
    """
    np.random.seed(0)
    deltas = [np.mean(np.random.choice(alt, len(alt))) - np.mean(np.random.choice(base, len(base)))
              for _ in range(n_boot)]
    lower = np.percentile(deltas, (100 - ci) / 2)
    upper = np.percentile(deltas, 100 - (100 - ci) / 2)
    return lower, upper


def get_data(top=10):
    """
    Load and combine data.

    :param top: what y's are considered to be 'high-rank'
    :return: combined dataframes
    """
    results_df = pd.read_csv(OUTPUT_DIR / f'y_{top}' / 'repeat_results.csv')
    gml_by_run = pd.read_csv(OUTPUT_DIR / f'y_{top}' / "gml_combined_runs.csv")
    gml_df = pd.read_csv(OUTPUT_DIR / f'y_{top}' / "gml_aggregates_ranked_by_pr_auc.csv")

    def summarize_by_class(df, class_str):
        df_filtered = df[df['target_class'] == class_str]
        group_cols = ['feature_set', 'model_name']
        numeric_cols = df_filtered.select_dtypes(include=np.number).columns
        mean_df = df_filtered.groupby(group_cols)[numeric_cols].mean().reset_index()
        std_df = df_filtered.groupby(group_cols)[numeric_cols].std().reset_index()

        # Merge mean and std with suffixes
        summary_df = pd.merge(mean_df, std_df, on=group_cols, suffixes=('_mean', '_std'))
        return summary_df

    # Tabular ML results
    df_high = summarize_by_class(results_df, 'high')
    df = df_high.sort_values(by='pr_auc_mean', ascending=False)

    combined = pd.concat([df, gml_df], ignore_index=True, sort=False)
    combined['target_class'] = combined['target_class'].fillna('N/A')
    combined['weighted'] = combined['weighted'].fillna('N/A')

    # Graph ML results
    gml_by_run = gml_by_run[gml_by_run['weighted'] == False]

    df_high_by_run = results_df[results_df['target_class'] == 'high']
    gml_high_by_run = gml_by_run[gml_by_run['target_class'] == 0]

    combined_by_run = pd.concat([df_high_by_run, gml_high_by_run], ignore_index=True, sort=False)
    combined_by_run['roc_auc'] = combined_by_run['roc_auc'].fillna('N/A')
    combined_by_run['weighted'] = combined_by_run['weighted'].fillna('N/A')

    return combined, combined_by_run


def get_bar_plot(df, random_guessing, avg_neighbor, from_phd, top=10):
    """
    Generates bar plots which appear in the paper.

    :param df: dataframe with results (pd.DataFrame)
    :param random_guessing: random guessing baseline (float)
    :param avg_neighbor: avg. neighbor rank baseline (float)
    :param from_phd: PhD rank baseline (float)
    :param top: what y's to consider as high-rank
    :return: None
    """

    max_models = 6

    x = np.arange(len(ordered_feature_sets))
    group_width = 0.8
    bar_width = group_width / max_models

    fig_width = 7.2  # in inches
    aspect_ratio = 8 / 15  # from original figsize
    fig_height = fig_width * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for j, feat in enumerate(ordered_feature_sets):
        model_data = df[df['feature_set'] == feat]
        models = [m for m in ordered_models if m in model_data['model_name'].values]

        group_center = x[j]
        n = len(models)
        start = group_center - (n - 1) * bar_width / 2

        for i, model in enumerate(models):
            row = model_data[model_data['model_name'] == model]
            if not row.empty:
                mean = row['pr_auc_mean'].values[0]
                std = row['pr_auc_std'].values[0]
                xpos = start + i * bar_width
                label = pretty_model_labels[model]
                ax.bar(
                    xpos, mean, width=bar_width, color=model_color_map[model],
                    yerr=std, capsize=3, alpha=0.9,
                    label=label if model not in ax.get_legend_handles_labels()[1] else None
                )

    # Horizontal baselines
    ax.axhline(random_guessing, color='black', linestyle=':', linewidth=2, label='Random Guessing')
    ax.axhline(avg_neighbor, color='black', linestyle='--', linewidth=1.5, label='By Avg. Co-author Rank')
    ax.axhline(from_phd, color='black', linestyle='-.', linewidth=1.5, label='By PhD Rank')

    # Text annotations
    if top == 10:
        arrow_xy = (0.7, 0.45)
        xytext = (0, 0.4)
        better_x = 0.38
        better_y = 0.435

        ax.text(2.6, 0.49, 'Avg. PR-AUC of 0.424', va='center', ha='right', fontsize=8)
        circle = patches.Ellipse(
            (2.3235, 0.424), width=0.3, height=0.035, transform=ax.transData, linewidth=1.5,
            edgecolor='black', facecolor='none', linestyle='-'
        )
        ax.add_patch(circle)
        ax.annotate(
            '',
            xy=(2.25, 0.435),
            xytext=(1.9, 0.48),
            arrowprops=dict(arrowstyle='->', color='black'),
            ha='center', va='center',
        )

        ax.text(6.2, 0.50, 'Avg. PR-AUC of 0.458', va='center', ha='right', fontsize=8)
        circle = patches.Ellipse(
            (5.9335, 0.458), width=0.3, height=0.035,  # same idea here
            transform=ax.transData,
            linewidth=1.5, edgecolor='black', facecolor='none', linestyle='-'
        )
        ax.add_patch(circle)
        ax.annotate(
            '',
            xy=(5.85, 0.465),
            xytext=(5.5, 0.49),
            arrowprops=dict(arrowstyle='->', color='black'),
            ha='center', va='center',
        )
    elif top == 20:
        arrow_xy = (0.5, 0.63)
        xytext = (0.5, 0.59)
        better_x = 0.45
        better_y = 0.61
    elif top == 30:
        arrow_xy = (0.5, 0.77)
        xytext = (0.5, 0.71)
        better_x = 0.45
        better_y = 0.74
    elif top == 40:
        arrow_xy = (0.6, 0.91)
        xytext = (0.6, 0.85)
        better_x = 0.55
        better_y = 0.88
    elif top == 50:
        arrow_xy = (0.6, 0.97)
        xytext = (0.6, 0.91)
        better_x = 0.55
        better_y = 0.94
    else:
        arrow_xy = (0.5, 0.49)
        xytext = (0.5, 0.45)
        better_x = 0.45
        better_y = 0.47

    ax.annotate(
        '',
        xy=arrow_xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle='->', color='black'),
        ha='center', va='center',
    )
    ax.text(better_x, better_y, 'Better', va='center', ha='right')

    ax.text(
        -0.02, 1.075,
        "Precision-Recall AUC by Model and Feature Set",
        fontweight='bold',
        ha='left',
        va='top',
        color='#333333',
        transform=ax.transAxes
    )

    # Formatting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_feature_labels[feat] for feat in ordered_feature_sets])
    ax.set_ylabel("PR-AUC")

    if top is None:
        ax.set_ylim([0.15, 0.5])
    elif top == 20:
        ax.set_ylim([0.15, 0.7])
    elif top == 30:
        ax.set_ylim([0.15, 0.85])
    elif top == 40 or top == 50:
        ax.set_ylim([0.15, 1.0])

    # Legend with fixed ordering
    handles, labels = ax.get_legend_handles_labels()
    pretty_order = ['Random Guessing', 'By Avg. Co-author Rank', 'By PhD Rank'] + [
        pretty_model_labels[m] for m in ordered_models if m in df['model_name'].values
    ]
    new_handles = [handles[labels.index(lbl)] for lbl in pretty_order if lbl in labels]

    fig.legend(
        new_handles, pretty_order,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=6,
        fontsize=8,
        title_fontsize=9,
        frameon=True
    )

    plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()
    ensure_dir(PLOTS_DIR)
    plt.savefig(PLOTS_DIR / f'barplot_y_{top}.png', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'barplot_y_{top}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'barplot_y_{top}.svg', dpi=600, bbox_inches='tight')

    plt.show()


def get_box_plot(df, random_guessing, avg_neighbor, from_phd, top=10):
    """
    Generates box plots which appear in the paper's appendix.

    :param df: dataframe with results (pd.DataFrame)
    :param random_guessing: random guessing baseline (float)
    :param avg_neighbor: avg. neighbor rank baseline (float)
    :param from_phd: PhD rank baseline (float)
    :param top: what y's to consider as high-rank
    :return: None
    """

    # Ensure consistent order
    df = df[df['feature_set'].isin(ordered_feature_sets)]
    df['feature_set'] = pd.Categorical(df['feature_set'], categories=ordered_feature_sets, ordered=True)

    fig_width = 7.2  # in inches
    aspect_ratio = 8 / 15  # from original figsize
    fig_height = fig_width * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Set up plot
    ax = sns.boxplot(
        data=df,
        x='feature_set',
        y='pr_auc',
        order=ordered_feature_sets,
        palette=feature_colors,
        width=0.6,
        fliersize=4
    )

    # Add horizontal baselines
    ax.axhline(random_guessing, color='black', linestyle=':', linewidth=2, label='Random Guessing')
    ax.axhline(avg_neighbor, color='black', linestyle='--', linewidth=1.5, label='By Avg. Co-author Rank')
    ax.axhline(from_phd, color='black', linestyle='-.', linewidth=1.5, label='By PhD Rank')

    # Formatting
    ax.set_xticklabels([pretty_feature_labels[feat] for feat in ordered_feature_sets])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")

    if top == 10:
        ax.set_ylim([0.15, 0.5])
    elif top == 20 or top == 30:
        ax.set_ylim([0.15, 0.85])
    elif top == 40 or top == 50:
        ax.set_ylim([0.15, 1.0])

    # add text annotations
    if top == 10:
        arrow_xy = (0.5, 0.485)
        xytext = (0.5, 0.46)
        better_x = 0.45
        better_y = 0.47

        ax.text(6.5, random_guessing - 0.005, 'Random Guessing', va='top', ha='right', style='italic')
        ax.text(6.5, avg_neighbor - 0.005, 'By Avg. Co-author Rank', va='top', ha='right', style='italic')
        ax.text(6.5, from_phd - 0.005, 'By PhD Rank', va='top', ha='right', style='italic')

    elif top == 20:
        arrow_xy = (0.5, 0.77)
        xytext = (0.5, 0.72)
        better_x = 0.45
        better_y = 0.74

        ax.text(6.5, random_guessing + 0.005, 'Random Guessing', va='bottom', ha='right', style='italic')
        ax.text(6.5, avg_neighbor - 0.005, 'By Avg. Co-author Rank', va='top', ha='right', style='italic')
        ax.text(6.5, from_phd + 0.005, 'By PhD Rank', va='bottom', ha='right', style='italic')

    elif top == 30:
        arrow_xy = (0.5, 0.77)
        xytext = (0.5, 0.72)
        better_x = 0.45
        better_y = 0.74

        ax.text(6.5, random_guessing - 0.005, 'Random Guessing', va='top', ha='right', style='italic')
        ax.text(6.5, avg_neighbor - 0.005, 'By Avg. Co-author Rank', va='top', ha='right', style='italic')
        ax.text(6.5, from_phd + 0.005, 'By PhD Rank', va='bottom', ha='right', style='italic')

    elif top == 40:
        arrow_xy = (0.6, 0.92)
        xytext = (0.6, 0.86)
        better_x = 0.55
        better_y = 0.88

        ax.text(6.49, random_guessing - 0.005, 'Random\nGuessing', va='top', ha='right', style='italic')
        ax.text(6.5, avg_neighbor - 0.005, 'By Avg. Co-author Rank', va='top', ha='right', style='italic')
        ax.text(6.5, from_phd + 0.005, 'By PhD Rank', va='bottom', ha='right', style='italic')

    elif top == 50:
        arrow_xy = (0.6, 0.98)
        xytext = (0.6, 0.91)
        better_x = 0.55
        better_y = 0.94

        ax.text(6.5, random_guessing - 0.005, 'Random Guessing', va='top', ha='right', style='italic')
        ax.text(6.5, avg_neighbor - 0.005, 'By Avg. Co-author Rank', va='top', ha='right', style='italic')
        ax.text(6.5, from_phd + 0.005, 'By PhD Rank', va='bottom', ha='right', style='italic')

    else:
        arrow_xy = (0.5, 0.48)
        xytext = (0.5, 0.46)
        better_x = 0.45
        better_y = 0.47

    ax.annotate(
        '',
        xy=arrow_xy,
        xytext=xytext,
        arrowprops=dict(arrowstyle='->', color='black'),
        ha='center', va='center',
    )
    ax.text(better_x, better_y, 'Better', va='center', ha='right')

    ax.text(
        -0.02, 1.075,
        "Precision-Recall AUC Distributions by Feature Set",
        fontweight='bold',
        ha='left',
        va='top',
        color='#333333',
        transform=ax.transAxes
    )

    # save figure
    plt.tight_layout()
    ensure_dir(PLOTS_DIR)
    plt.savefig(PLOTS_DIR / f'boxplot_y_{top}.png', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'boxplot_y_{top}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'boxplot_y_{top}.svg', dpi=600, bbox_inches='tight')

    plt.show()


def summary_line_plot(dfs):
    """
    Generates line plot figure in the paper.

    :param dfs: dataframes with results
    :return: None
    """
    thresholds = [10, 20, 30, 40, 50]

    # Set up comparisons, labels, colors, and markers
    comparisons_main = [
        ('cv', 'cv+graph'),
        ('biblio', 'biblio+graph'),
        ('cv+biblio', 'cv+biblio+graph')
    ]
    labels_main = {
        ('cv', 'cv+graph'): 'PhD+Co-author vs. PhD',
        ('biblio', 'biblio+graph'): 'Bib+Co-author vs. Bib',
        ('cv+biblio', 'cv+biblio+graph'): 'PhD+Bib+Co-author vs. PhD+Bib'
    }
    colors_main = ["#b6655e", "#aa7d17", "#427073"]
    markers_main = ['o', 's', '^']
    valid_sets_main = ['cv', 'biblio', 'cv+graph', 'biblio+graph', 'cv+biblio', 'cv+biblio+graph']

    comparisons_alt = [
        ('biblio', 'cv'),
        ('biblio+graph', 'cv+graph')
    ]
    labels_alt = {
        ('biblio', 'cv'): 'PhD vs. Bib',
        ('biblio+graph', 'cv+graph'): 'PhD+Co-author vs. Bib+Co-author',
    }
    colors_alt = ["#984c88", "#6c3d6e"]
    markers_alt = ['o', 's']
    valid_sets_alt = ['cv', 'biblio', 'cv+graph', 'biblio+graph']

    # Calculate the differences in PR-AUC between comparison feature sets
    def compute_deltas(dfs, comparisons, valid_sets):
        delta_pr_auc = {label: [] for label in comparisons}
        ci_lowers = {label: [] for label in comparisons}
        ci_uppers = {label: [] for label in comparisons}

        for top in thresholds:
            df = dfs[top]
            df = df[df['feature_set'].isin(valid_sets)]
            for (base, other) in comparisons:
                base_vals = df[df['feature_set'] == base]['pr_auc']
                other_vals = df[df['feature_set'] == other]['pr_auc']

                if base_vals.empty or other_vals.empty:
                    delta_pr_auc[(base, other)].append(np.nan)
                    ci_lowers[(base, other)].append(np.nan)
                    ci_uppers[(base, other)].append(np.nan)
                else:
                    delta = other_vals.mean() - base_vals.mean()
                    lower, upper = ci_bounds_differential(base_vals, other_vals)
                    delta_pr_auc[(base, other)].append(delta)
                    ci_lowers[(base, other)].append(lower)
                    ci_uppers[(base, other)].append(upper)
        return delta_pr_auc, ci_lowers, ci_uppers

    deltas_main_no, lows_main_no, highs_main_no = compute_deltas(dfs, comparisons_main, valid_sets_main)
    deltas_alt_no, lows_alt_no, highs_alt_no = compute_deltas(dfs, comparisons_alt, valid_sets_alt)

    # Set up figure
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = GridSpec(3, 2, height_ratios=[0.05, 1, 0.25], hspace=0.75, wspace=0.4)
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # Plot the differential between different features and annotate with a * if it's significantly different
    def plot_delta(ax, deltas, lowers, uppers, comparisons, labels, markers, colors, significance_flags):
        for i, comp in enumerate(comparisons):
            means = deltas[comp]
            lows = lowers[comp]
            highs = uppers[comp]

            if np.isnan(means).all():
                continue

            ax.plot(thresholds, means, label=labels[comp], marker=markers[i], color=colors[i])
            ax.fill_between(thresholds, lows, highs, color=colors[i], alpha=0.2)

            for j, t in enumerate(thresholds):
                if significance_flags.get(comp, [0]*len(thresholds))[j]:
                    ax.annotate('*', (t, means[j] + 0.001), ha='center', va='bottom', fontsize=10)

        ax.axhline(0, linestyle='--', color='gray', linewidth=1)
        ax.set_xlabel("Top-X Threshold")
        ax.set_ylabel(r"$\Delta$ PR-AUC")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Plot (significance flags from stats results)
    plot_delta(axs[0], deltas_main_no, lows_main_no, highs_main_no, comparisons_main, labels_main, markers_main, colors_main,
               {
                   ('cv', 'cv+graph'): [1,0,0,0,0],
                   ('biblio', 'biblio+graph'): [1,0,1,0,0],
                   ('cv+biblio', 'cv+biblio+graph'): [1,0,0,0,0]
               })
    plot_delta(axs[1], deltas_alt_no, lows_alt_no, highs_alt_no, comparisons_alt, labels_alt, markers_alt, colors_alt,
               {
                   ('biblio', 'cv'): [1,0,1,0,0],
                   ('biblio+graph', 'cv+graph'): [0,0,1,0,0],
               })

    axs[0].set_ylim([-0.03, 0.125])
    axs[1].set_ylim([-0.075, 0.08])

    # Titles
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(-0.115, 0, 'Performance differences between feature sets for various definitions of "high-rank"',
                  fontsize=10, fontweight='bold', ha='left', va='top', color='#333333',
                  transform=title_ax.transAxes)

    panel_labels = ['(a)', '(b)']
    for ax, label in zip(axs, panel_labels):
        ax.text(-0.2, 1.15, label, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='left', color='#444444')

    axs[0].text(-0.1, 1.08, "Adding Co-author Features", fontweight='bold', fontsize=9, color='#444444', transform=axs[0].transAxes)
    axs[1].text(-0.1, 1.08, "PhD Rank vs. Bibliometric Features", fontweight='bold', fontsize=9, color='#444444', transform=axs[1].transAxes)

    # Annotations
    axs[0].annotate(
        '',
        xy=(9, 0.078),
        xytext=(9, 0.053),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
        # fontsize=8
    )
    axs[0].text(10, 0.063, 'Better with\nCo-author Feats.', va='center', ha='left', fontsize=6, color='#666666')

    axs[0].annotate(
        '',
        xy=(9, -0.025),
        xytext=(9, -0.001),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
    )
    axs[0].text(10, -0.014, 'Worse with\nCo-author Feats.', va='center', ha='left', fontsize=6, color='#666666')

    axs[0].annotate(
        '',
        xy=(31, 0.075),
        xytext=(40, 0.088),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
        # fontsize=8
    )
    axs[0].text(44, 0.092, 'Significant', va='top', ha='center', fontsize=6, color='#666666')

    axs[1].annotate(
        '',
        xy=(9, 0.035),
        xytext=(9, 0.005),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
        # fontsize=8
    )
    axs[1].text(10, 0.018, 'PhD Rank$>$\nBib Feats.', va='center', ha='left', fontsize=6, color='#666666')

    axs[1].annotate(
        '',
        xy=(24, -0.035),
        xytext=(24, -0.005),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
        # fontsize=8
    )
    axs[1].text(25, -0.018, 'Bib Feats.$>$\nPhD Rank', va='center', ha='left', fontsize=6, color='#666666')

    axs[1].annotate(
        '',
        xy=(31, 0.067),
        xytext=(39, 0.073),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center',
        # fontsize=8
    )
    axs[1].text(43, 0.078, 'Significant', va='top', ha='center', fontsize=6, color='#666666')

    # Legend
    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')
    legend_elements = [
        Patch(facecolor=colors_main[0], edgecolor=colors_main[0], label=labels_main[('cv', 'cv+graph')]),
        Patch(facecolor=colors_main[2], edgecolor=colors_main[2], label=labels_main[('cv+biblio', 'cv+biblio+graph')]),
        Patch(facecolor=colors_main[1], edgecolor=colors_main[1], label=labels_main[('biblio', 'biblio+graph')]),
        Patch(facecolor=colors_alt[0], edgecolor=colors_alt[0], label=labels_alt[('biblio', 'cv')]),
        Patch(facecolor=colors_alt[1], edgecolor=colors_alt[1], label=labels_alt[('biblio+graph', 'cv+graph')]),
    ]
    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        fontsize=7,
        bbox_to_anchor=(0.46, 0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.05)
    ensure_dir(PLOTS_DIR)
    plt.savefig(PLOTS_DIR / f'lineplot.png', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'lineplot.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'lineplot.svg', dpi=600, bbox_inches='tight')

    plt.show()


def rewire(random_guessing, avg_neighbor, from_phd):
    """
    Generates rewired plot that appears in the SI.

    :param random_guessing: Random guessing baseline (float)
    :param avg_neighbor: Avg. neighbor rank baseline (float)
    :param from_phd: PhD rank baseline (float)
    :return: None
    """
    # Load data
    df = pd.read_csv(REWIRE_DIR / 'rewired_no_graph_feats_summary.csv')
    df['percent'] = df['percent'].astype(int)

    baseline_df = pd.read_csv(OUTPUT_DIR / f'y_10' / "gml_combined_runs.csv")
    baseline_df = baseline_df[baseline_df['target_class'] == 0]

    # Set up annotations
    titles = ['PhD  + Co-author', 'Bib + Co-author', 'PhD + Bib + Co-author']
    feature_sets = ['cv', 'biblio', 'cv+biblio']
    abc = ['(a)', '(b)', '(c)']
    models = ['GCN', 'GAT', 'GraphSAGE', 'GConvGRU']

    feature_map = {
        'cv+graph': 'cv',
        'biblio+graph': 'biblio',
        'cv+biblio+graph': 'cv+biblio'
    }

    baseline_df['feature_set'] = baseline_df['feature_set'].map(feature_map)

    coordinates = [(1, 0), (1, 1), (1, 2)]

    # Set up figure
    figsize = (7.2, 3.25)
    figure_nrows = 3  # Header, 1 row of plots, legend
    figure_ncols = 3
    figure_hspace = 0.6
    figure_wspace = 0.4
    figure_height_ratios = [0.1, 1.0, 0.2]
    figure_width_ratios = [1.0, 1.0, 1.0]
    dpi = 600

    xlabel = '% Rewired'
    ylabel = 'PR-AUC'
    ticklabelsize = 7
    label_fontsize = 8

    header = 'Effect of Degree-Preserving Rewiring on Model Performance'
    header_coords = (-0.10, 0.4)
    header_style = {'fontsize': 10, 'fontcolor': '#333333', 'fontweight': 'bold'}

    fig, gs = create_figure(
        figsize=figsize,
        nrows=figure_nrows,
        ncols=figure_ncols,
        height_ratios=figure_height_ratios,
        width_ratios=figure_width_ratios,
        hspace=figure_hspace,
        wspace=figure_wspace
    )

    fig = add_text(fig, gs[0, :], header, header_coords, header_style)

    # Generate subplots
    for i, feature in enumerate(feature_sets):
        ax = fig.add_subplot(gs[coordinates[i]])

        for model in models:
            subset = df[(df['feature_set'] == feature) & (df['model_name'] == model)]

            def compute_ci(group, ci=90):
                pr_aucs = group['pr_auc'].values
                mean = np.mean(pr_aucs)
                lower, upper = ci_bounds(pr_aucs, ci=ci)
                return pd.Series({'mean': mean, 'lower': lower, 'upper': upper})

            grouped = subset.groupby('percent').apply(compute_ci).reset_index()

            baseline_subset = baseline_df[
                (baseline_df['model_name'] == model) &
                (baseline_df['feature_set'] == feature)
                ]

            if not baseline_subset.empty:
                pr_auc_0 = baseline_subset['pr_auc'].mean()

                # Create the baseline row at percent=0
                baseline_row = pd.DataFrame({
                    'percent': [0],
                    'mean': [pr_auc_0],
                    'lower': [pr_auc_0],
                    'upper': [pr_auc_0]
                })

                # Add to grouped and sort
                grouped = pd.concat([baseline_row, grouped], ignore_index=True)
                grouped = grouped.sort_values('percent').reset_index(drop=True)

            ax.plot(grouped['percent'], grouped['mean'], label=model, color=model_color_map[model], marker='o',
                    markersize=3, linewidth=1)
            ax.fill_between(grouped['percent'],
                            grouped['lower'],
                            grouped['upper'],
                            color=model_color_map[model], alpha=0.2)

        ax.axhline(random_guessing, color='black', linestyle=':', linewidth=1.5, label='Random Guessing')
        ax.axhline(avg_neighbor, color='black', linestyle='--', linewidth=1, label='By Avg. Co-author Rank')
        ax.axhline(from_phd, color='black', linestyle='-.', linewidth=1, label='By PhD Rank')

        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)

        ax.set_ylim([0.2, 0.5])
        ax.set_xlim([-1, 100])
        ax.tick_params(labelsize=ticklabelsize)

        ax.text(-0.2, 1.1, abc[i], transform=ax.transAxes,
                fontsize=10, fontweight='bold', color='#444444')
        ax.text(-0.05, 1.09, titles[i], transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='#444444')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # Legend with fixed ordering
    handles, labels = ax.get_legend_handles_labels()
    pretty_order = ['Random Guessing', 'By Avg. Co-author Rank', 'By PhD Rank'] + models
    new_handles = [handles[labels.index(lbl)] for lbl in pretty_order if lbl in labels]

    fig.legend(
        new_handles, pretty_order,
        loc='lower center',
        ncol=3,
        fontsize=7,
        title_fontsize=9,
        frameon=True
    )

    ensure_dir(PLOTS_DIR)
    plt.savefig(PLOTS_DIR / f'rewiring.png', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'rewiring.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / f'rewiring.svg', dpi=600, bbox_inches='tight')

    plt.show()


def main():
    """
    Generates the plots present in the paper.

    :return: None
    """
    tops = [10, 20, 30, 40, 50]

    dfs = {}
    for top in tops:
        df, df_by_run = get_data(top)
        dfs[top] = df_by_run

        random_guessing = explicit_baselines.get_random_guessing(top)
        avg_neighbor = explicit_baselines.predict_from_avg_neighbor_rank(top)
        from_phd = explicit_baselines.predict_from_doc_ranking(top)

        # Generate bar chart figure (Figures 3, A3, A5, A7, A9)
        get_bar_plot(df, random_guessing, avg_neighbor, from_phd, top=top)

        # Generate bar chart figure (Figures A2, A4, A6, A8, A10)
        get_box_plot(df_by_run, random_guessing, avg_neighbor, from_phd, top=top)

        if top == 10:
            # Generate rewiring figure (Figure A1)
            rewire(random_guessing, avg_neighbor, from_phd)

    # Generate summary line plots figure (Figure 4)
    summary_line_plot(dfs)


if __name__=='__main__':
    main()