"""
infer_gender.py

Infers gender of faculty members using inferred labels from SOTA proprietary APIs,
    (1) Gender-API - gender-api.com
    (2) genderize.io - genderize.io
Final gender labels are assigned based on the following decision rules:
    If Gender-API and genderize.io both have confidence >= 0.8 and agree, use that gender label.
    Else, manually label.

Outputs:
- '../../data/hiring/names_gendered_with_auto_label.csv'
- '../../data/hiring/ambiguous_gender.csv' --> this must be manually labeled
- '../../data/hiring/faculty.csv'

12/5/2025 - SD
"""

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# API keys
GENDER_API_KEY = "put key here"
GENDERIZE_IO_KEY = "put key here"


def split_author(name):
    """
    Split a full name into first and last name, skipping initials.

    :param name: full name to split
    :return: the first and last name
    """

    if not isinstance(name, str):
        return "", ""

    parts = name.split()
    if len(parts) == 0:
        return "", ""

    # skip initial-only parts
    i = 0
    while i < len(parts) and len(parts[i]) == 1:
        i += 1

    if i < len(parts):
        first_name = parts[i]
        last_name = " ".join(parts[i + 1:])
    else:
        first_name = None
        last_name = " ".join(parts)

    return first_name, last_name


def gender_api(name, api_key):
    """
    Call GenderAPI and label name.

    :param name: name to gender label
    :param api_key: api key for Gender-API.com
    :return: a dictionary with inferred gender information
    """

    url = "https://gender-api.com/get"
    params = {"name": name, "key": api_key}

    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"[GenderAPI] Failed for '{name}' (status {resp.status_code})")
        return None


def genderize_io(name, api_key):
    """
    Call genderize.io.

    :param name: name to gender label
    :param api_key: api key for genderize.io
    :return: a dictionary with inferred gender information
    """

    url = "https://api.genderize.io/"
    params = {"name": name, "apikey": api_key}

    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"[Genderize.io] Failed for '{name}' (status {resp.status_code})")
        return None


def get_gender(df, name_col, gender_api_key, genderize_io_key):
    """
    Loop through df[name_col], call Gender-API / genderize.io, and
    add columns with results. Returns modified df.

    :param df: dataframe with names to gender label
    :param name_col: column which stores the names
    :param gender_api_key: api key for Gender-API
    :param genderize_io_key: api key for genderize.io
    :return: the dataframe with gender information
    """
    # Initialize columns
    df["genderapi_gender"] = "unknown"
    df["genderapi_p"] = np.nan
    df["genderapi_n"] = np.nan

    df["genderizeio_gender"] = "unknown"
    df["genderizeio_p"] = np.nan
    df["genderizeio_n"] = np.nan

    use_gender_api = gender_api_key and gender_api_key.lower() != "skip"
    use_genderize = genderize_io_key and genderize_io_key.lower() != "skip"

    if not use_gender_api and not use_genderize:
        print("[Info] Both API keys set to 'skip'; assigning 'unknown' for all.")
        return df

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gender labeling"):
        name = row[name_col]

        if not isinstance(name, str) or name.strip() == "":
            continue

        # GenderAPI
        if use_gender_api:
            try:
                result = gender_api(name, gender_api_key)
                if result:
                    df.at[idx, "genderapi_gender"] = result.get("gender", "unknown")
                    df.at[idx, "genderapi_p"] = result.get("accuracy", 0) / 100.0
                    df.at[idx, "genderapi_n"] = result.get("samples", 0)
            except Exception as e:
                print(f"[GenderAPI] Error on '{name}': {e}")

        # Genderize.io
        if use_genderize:
            try:
                result = genderize_io(name, genderize_io_key)
                if result:
                    gender = result.get("gender") or "unknown"
                    df.at[idx, "genderizeio_gender"] = gender
                    df.at[idx, "genderizeio_p"] = result.get("probability", 0.0)
                    df.at[idx, "genderizeio_n"] = result.get("count", 0)
            except Exception as e:
                print(f"[Genderize.io] Error on '{name}': {e}")

    return df


def get_genderapi_genderizeio_labels(faculty_df):
    """
    Label the names with gender information from Gender-API and genderize.io.

    :param faculty_df: faculty-level dataframe
    :return: dataframe of names with gender information
    """
    df = faculty_df['author'].copy()

    # Detect / construct first_name column
    if "first_name" in df.columns:
        name_col = "first_name"
    elif "name" in df.columns:
        df[["first_name", "last_name"]] = df["name"].apply(
            lambda x: pd.Series(split_author(x))
        )
        name_col = "first_name"
    elif "author" in df.columns:
        df[["first_name", "last_name"]] = df["author"].apply(
            lambda x: pd.Series(split_author(x))
        )
        name_col = "first_name"
    else:
        raise ValueError(
            "Could not find a name column. "
            "Expected one of: 'first_name', 'name', or 'author'."
        )

    print(f"[Info] Using '{name_col}' as the first-name column for gender labeling.")

    # Label only unique names
    unique_names = df[[name_col]].dropna().drop_duplicates().reset_index(drop=True)
    unique_labeled = get_gender(unique_names.copy(), name_col, GENDER_API_KEY, GENDERIZE_IO_KEY)

    # Merge labeled names back onto full df
    df = df.merge(unique_labeled, on=name_col, how="left")

    return df


def assign_gender(df, faculty_df):
    """
    Assign a single gender label based on Gender-API and genderize.io info, and decision rules.

    :param df: dataframe with gender information
    :param faculty_df: full faculty-level dataframe
    :return: df - full dataframe with final gender labels; ambiguous_df - dataframe with ambiguous names
    """
    for col in ["genderapi_p", "genderizeio_p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean / normalize gender strings and probabilities
    df["genderapi_gender_clean"] = df["genderapi_gender"].str.lower().str.strip()
    df["genderizeio_gender_clean"] = df["genderizeio_gender"].str.lower().str.strip()

    # Define "high confidence" and "valid" predictions for each tool
    threshold = 0.8

    gapi_valid = (
            df["genderapi_gender_clean"].isin(["male", "female"])
            & (df["genderapi_p"] >= threshold)
    )

    gz_valid = (
            df["genderizeio_gender_clean"].isin(["male", "female"])
            & (df["genderizeio_p"] >= threshold)
    )

    # Require both tools to be valid AND to agree
    agree = df["genderapi_gender_clean"] == df["genderizeio_gender_clean"]
    auto_ok = gapi_valid & gz_valid & agree

    # Create a new automatic label column using the consensus rule
    df["gender_label_auto"] = pd.NA
    df.loc[auto_ok, "gender_label_auto"] = df.loc[auto_ok, "genderapi_gender_clean"]

    # Save ambiguous / manual-review cases
    ambiguous_df = df[~auto_ok].copy()

    faculty_small = faculty_df[["author", "university"]].copy()
    ambiguous_df = ambiguous_df.merge(faculty_small, on="author", how="left")
    cols = list(ambiguous_df.columns)
    first_cols = ["author", "university"]
    remaining_cols = [c for c in cols if c not in first_cols]
    ambiguous_df = ambiguous_df[first_cols + remaining_cols]

    return df, ambiguous_df


def combine_full_gender(faculty_df, automatic, manual):
    """
    Merge gender information (both automatically and manually determined) back into
    full faculty dataframe.

    :param faculty_df: full faculty-level dataframe
    :param automatic: dataframe with automatically-labele gender
    :param manual: dataframe with manually-labeled gender
    :return: faculty-level dataframe with gender information
    """

    auto_gender = (
        automatic[["author", "gender_label_auto"]]
        .dropna(subset=["gender_label_auto"])
        .drop_duplicates(subset=["author"])
        .assign(source="auto")
    )

    manual_gender = (
        manual[["author", "gender_label_auto"]]
        .dropna(subset=["gender_label_auto"])
        .drop_duplicates(subset=["author"])
        .assign(source="manual")
    )

    gender_map = pd.concat([auto_gender, manual_gender], ignore_index=True)

    gender_map = (
        gender_map.sort_values(by=["author", "source"])
        .drop_duplicates(subset=["author"], keep="last")
        .drop(columns=["source"])
        .rename(columns={"gender_label_auto": "gender_string"})
    )

    faculty_df = faculty_df.merge(gender_map, on="author", how="left")

    missing = faculty_df["gender_string"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} authors still have no gender_string after merge.")

    gender_map_numeric = {"male": 0, "female": 1}
    faculty_df["gender"] = faculty_df["gender_string"].map(gender_map_numeric)

    missing_numeric = faculty_df["gender"].isna().sum()
    if missing_numeric > 0:
        print(f"Warning: {missing_numeric} authors have non-binary/unknown gender_string.")

    return faculty_df


def main():

    faculty_df = pd.read_csv('../../data/hiring/faculty.csv')

    # get gender labels from Gender-API and genderize.io
    gendered_df = get_genderapi_genderizeio_labels(faculty_df)

    # assign a final gender label based on decision rules
    #   if Gender-API and genderize.io agree, and both have confidence >= 0.8, use that label
    #   else manually assign
    auto_gendered_df, ambiguous_gender_df = assign_gender(gendered_df, faculty_df)

    auto_gendered_df.to_csv("../../data/hiring/names_gendered_with_auto_label.csv", index=False)
    ambiguous_gender_df.to_csv("../../data/hiring/ambiguous_gender.csv", index=False)

    # Note: must manually add gender to all rows in ambiguous_gender_df before running final combine_full_gender()

    auto_gendered_df = pd.read_csv("../../data/hiring/names_gendered_with_auto_label.csv")
    manually_gendered_df = pd.read_csv("../../data/hiring/ambiguous_gender.csv")
    faculty_df = combine_full_gender(faculty_df, auto_gendered_df, manually_gendered_df)

    faculty_df.to_csv("../../data/hiring/faculty.csv", index=False)

if __name__ == "__main__":
    main()
