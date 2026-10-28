#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_object_dtype

import json

def preprocess_csv(input_csv, skip_columns, extra_category_columns, missing_value):
    skip_cols = [c.strip() for c in skip_columns.split(",") if c.strip()]
    extra_category_cols = [c.strip() for c in extra_category_columns.split(",") if c.strip()]

    # Read original for classification
    df_original = pd.read_csv(input_csv)

    # Replace user-defined missing markers with NaN
    if missing_value is not None:
        df_original.replace(missing_value, np.nan, inplace=True)

    # Detect numeric columns BEFORE encoding
    numerical_cols = [col for col in df_original.columns 
                      if is_numeric_dtype(df_original[col]) and col not in skip_cols]

    # Everything else → categorical
    category_cols = [col for col in df_original.columns 
                     if col not in skip_cols and col not in numerical_cols]

    # Merge extra categories
    category_cols = list(set(category_cols + extra_category_cols))

    # Unhandled columns → skipped
    unhandled_cols = skip_cols

    # Save column info BEFORE processing
    column_info = {
        "category_columns": category_cols,
        "numerical_columns": numerical_cols,
        "unhandled_columns": unhandled_cols
    }
    info_path = os.path.join(os.path.dirname(input_csv), "column_info.json")
    with open(info_path, "w") as f:
        json.dump(column_info, f, indent=4)
    print(f"📄 Column info saved to {info_path}")

    # Process a copy for missing handling
    df_processed = df_original.copy()
    df_processed = handle_missing_auto(df_processed, skip_cols, extra_category_cols)

    # Round float columns to 4 decimal places
    for col in df_processed.columns:
        if col not in skip_cols and is_numeric_dtype(df_processed[col]) and df_processed[col].dtype.kind == 'f':
            df_processed[col] = df_processed[col].round(4)

    # Save processed file
    output_path = os.path.join(os.path.dirname(input_csv), "processed.csv")
    df_processed.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

    # Print summary based on ORIGINAL classification
    print("\nProcessing Summary:")
    for col in df_original.columns:
        if col in skip_cols:
            print(f"  {col:<20} → skipped")
        elif col in category_cols:
            print(f"  {col:<20} → categorical")
        elif col in numerical_cols:
            print(f"  {col:<20} → numeric")


def handle_missing_auto(df, skip_cols, extra_category_cols):
    """Automatically handle missing values and convert categorical/boolean to numeric."""
    # Detect categorical columns from Pandas
    detected_cats = [col for col in df.columns if is_object_dtype(df[col]) or df[col].dtype.name == "category"]
    # Merge with extra categories provided by user
    category_cols = list(set(detected_cats + extra_category_cols))

    for col in df.columns:
        if col in skip_cols:
            continue  # Skip processing for these columns

        missing_ratio = df[col].isna().mean()

        # Handle categorical columns
        if col in category_cols:
            existing_vals = df[col].dropna().unique()
            if len(existing_vals) > 0:
                df[col] = df[col].apply(
                    lambda x: np.random.choice(existing_vals) if pd.isna(x) else x
                )
            df[col] = pd.Categorical(df[col]).codes
            continue

        # Boolean columns
        if is_bool_dtype(df[col]):
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val).astype(int)

        # Numeric columns
        elif is_numeric_dtype(df[col]):
            if missing_ratio < 0.2:
                median_val = df[col].median(skipna=True)
                df[col] = df[col].fillna(median_val)
            else:
                col_min = df[col].min(skipna=True)
                col_max = df[col].max(skipna=True)
                df[col] = df[col].apply(
                    lambda x: np.random.uniform(col_min, col_max) if pd.isna(x) else x
                )

        else:
            # Fallback: treat as categorical if unknown dtype
            existing_vals = df[col].dropna().unique()
            if len(existing_vals) > 0:
                df[col] = df[col].apply(
                    lambda x: np.random.choice(existing_vals) if pd.isna(x) else x
                )
            df[col] = pd.Categorical(df[col]).codes

    return df


def main():
    parser = argparse.ArgumentParser(
        description="CSV preprocessing utility: auto missing handling, categorical/boolean to numeric, float rounding"
    )

    # 1. input_csv as first positional parameter (not optional)
    parser.add_argument("input_csv", help="Path to input CSV file")

    # 2. --skip-columns with shorthand -s
    parser.add_argument(
        "-s", "--skip-columns",
        default="",
        help="Comma-separated column names to skip processing"
    )

    # 3. --extra-category with shorthand -e
    parser.add_argument(
        "-e", "--extra-category",
        default="",
        help="Comma-separated column names to treat as categorical in addition to Pandas detection"
    )

    # 4. --missing-value with shorthand -m
    parser.add_argument(
        "-m", "--missing-value",
        default=None,
        help="Representation of missing values (e.g., 'NA', '?', '')"
    )

    args = parser.parse_args()

    preprocess_csv(
        args.input_csv,
        args.skip_columns,
        args.extra_category,
        args.missing_value
    )


if __name__ == "__main__":
    main()
















