#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def combine_csv_files(file1: str, file2: str):
    """
    Combine two CSV files side-by-side, skipping duplicate columns from the second file.
    Output is saved as 'combined_csv.csv' in the same directory as file1.
    """
    # Read CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Check row count
    if len(df1) != len(df2):
        raise ValueError(f"Row count mismatch: {len(df1)} vs {len(df2)}")

    # Remove duplicate columns from df2
    duplicate_cols = [col for col in df2.columns if col in df1.columns]
    if duplicate_cols:
        print(f"⚠ Skipping duplicate columns from second file: {duplicate_cols}")
        df2 = df2.drop(columns=duplicate_cols)

    # Combine side-by-side
    combined_df = pd.concat([df1, df2], axis=1)

    # Output path
    output_file = os.path.join(os.path.dirname(file1), "combined_csv.csv")

    # Save combined CSV
    combined_df.to_csv(output_file, index=False)
    print(f"✅ Combined CSV saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine two CSV files side-by-side, skipping duplicate columns from the second file."
    )
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")

    args = parser.parse_args()

    try:
        combine_csv_files(args.file1, args.file2)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
