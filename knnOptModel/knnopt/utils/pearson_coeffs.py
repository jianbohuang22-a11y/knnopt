#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def compute_correlations(csv_path, targets_str, skip_columns_str=None):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Parse targets list
    targets = [t.strip() for t in targets_str.split(",") if t.strip()]
    
    # Validate targets
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Target columns not found in CSV: {missing_targets}")

    # Parse skip columns
    skip_columns = [s.strip() for s in skip_columns_str.split(",")] if skip_columns_str else []
    skip_columns = [col for col in skip_columns if col]  # remove empty strings

    # Features = all columns not in targets and not in skip_columns
    features = [col for col in df.columns if col not in targets and col not in skip_columns]

    # Compute Pearson correlations
    corr_df = pd.DataFrame(index=targets, columns=features, dtype=float)
    for target in targets:
        for feature in features:
            corr_df.loc[target, feature] = df[target].corr(df[feature])

    # Save raw correlation coefficients
    output_path = os.path.join(os.path.dirname(csv_path), "correlation_coefficients.csv")
    corr_df.to_csv(output_path)
    print(f"✅ Correlation coefficients saved to {output_path}")

    # Normalize coefficients target-wise
    normalized_corr = corr_df.copy()
    for target in targets:
        max_abs_val = normalized_corr.loc[target].abs().max()
        if max_abs_val > 0:
            normalized_corr.loc[target] = normalized_corr.loc[target] / max_abs_val
        else:
            normalized_corr.loc[target] = 0.0  # avoid division by zero

    norm_output_path = os.path.join(os.path.dirname(csv_path), "normalized_coefficients.csv")
    normalized_corr.to_csv(norm_output_path)
    print(f"✅ Normalized coefficients saved to {norm_output_path}")


    print(f"📊 Shape: {corr_df.shape}")
    print(f"🔍 Targets: {targets}")
    print(f"🔍 Features: {features}")
    print(f"🚫 Skipped columns: {skip_columns}")

def main():
    parser = argparse.ArgumentParser(description="Compute Pearson correlation coefficients between features and targets, with normalization and transformation.")
    parser.add_argument("csv_path", help="Path to input CSV file")
    parser.add_argument("targets", help="Comma-separated list of target columns")
    parser.add_argument("-s", "--skip_columns", default=None, help="Comma-separated list of columns to skip from correlation")
    args = parser.parse_args()

    compute_correlations(args.csv_path, args.targets, args.skip_columns)

if __name__ == "__main__":
    main()
