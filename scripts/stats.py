"""
stats.py

This script conducts a Linear Mixed Effects statistical analysis for numerous baselines.

10/29/2025 - SD
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


def ensure_dir(p):
    """
    Ensures that files exist.

    :param p: filepath to check
    :return: None
    """
    Path(p).mkdir(parents=True, exist_ok=True)


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


def set_reference_level(df, ref_feature):
    """
    Updates the reference feature for Linear Mixed Effects modeling.

    :param df: dataframe of results
    :param ref_feature: Reference feature
    :return: Dataframe with updated order which puts reference feature first
    """
    df = df.copy()
    df['feature_set'] = df['feature_set'].astype('category')

    new_order = [ref_feature] + [x for x in df['feature_set'].cat.categories if x != ref_feature]
    df['feature_set'] = df['feature_set'].cat.reorder_categories(new_order, ordered=True)

    return df


def run_all_reference_models(df, top=10):
    """
    Runs Linear Mixed Effects models for various feature set baselines.

    :param df: dataframe of results
    :param top: which y's to consider 'high-rank'
    :return: None
    """
    reference_groups = ["cv", "biblio", "cv+biblio", "cv+graph"]

    for ref in reference_groups:
        df_ref = set_reference_level(df, ref)

        # Custom filename per reference group
        filename = f"mixed_effects_top{top if top else 'all'}"
        filename += f"_ref-{ref}.txt"

        # Ensure clean types
        df_ref = df_ref.copy()
        df_ref['model_name'] = df_ref['model_name'].astype('category')
        df_ref['feature_set'] = df_ref['feature_set'].astype('category')
        df_ref = df_ref.dropna(subset=['pr_auc'])

        # Fit the model
        model = smf.mixedlm("pr_auc ~ feature_set", df_ref, groups=df_ref["model_name"])
        result = model.fit()

        # Save to file
        ensure_dir(OUTPUT_DIR / 'stats')
        with open(OUTPUT_DIR / 'stats' / filename, "w") as f:
            f.write(result.summary().as_text())

        print(result.summary())

        print(f"\nMixed-effects model with reference '{ref}' written to {OUTPUT_DIR / 'stats' / filename}")


def main():
    """
    Runs the Linear Mixed Effects statistical analysis.

    :return: None
    """
    tops = [10, 20, 30, 40, 50]

    for top in tops:
        _, df_by_run = get_data(top)

        # Generate stats results for the paper (Tables 3, A1, A6-A9)
        run_all_reference_models(df_by_run, top)

if __name__=='__main__':
    main()