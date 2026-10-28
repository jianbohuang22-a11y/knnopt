#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import os

def compare_csv(file1: str, file2: str, columns: str = None, eps: float = 1e-8):
    """
    Compare two CSV files cell-wise and output differences.

    Parameters:
    - file1: Path to first CSV
    - file2: Path to second CSV
    - eps: Small constant to avoid division by zero
    - columns: Optional comma-separated list of column names to compare.
               If provided, only these columns are compared, and shape consistency is not enforced.
               Both files must contain these columns with the same number of rows.
    """
    # Read CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    if columns:
        # Split comma-separated string into list
        col_list = [c.strip() for c in columns.split(",") if c.strip()]
        
        # Check that both files contain these columns
        missing1 = [c for c in col_list if c not in df1.columns]
        missing2 = [c for c in col_list if c not in df2.columns]
        if missing1:
            raise ValueError(f"{file1} is missing columns: {missing1}")
        if missing2:
            raise ValueError(f"{file2} is missing columns: {missing2}")

        # Select only these columns
        df1 = df1[col_list]
        df2 = df2[col_list]

        # Check row count consistency for selected columns
        if len(df1) != len(df2):
            raise ValueError(f"Selected columns must have same number of rows. Got {len(df1)} vs {len(df2)}")
    else:
        # Original shape validation
        if df1.shape != df2.shape:
            raise ValueError(f"CSV files must have the same number of rows and columns. Got {df1.shape} vs {df2.shape}")

        # Check headers
        if set(df1.columns) != set(df2.columns):
            raise ValueError("CSV files must have identical headers (names), even if order differs.")

        # If order differs, reorder df2 to match df1
        if list(df1.columns) != list(df2.columns):
            print("⚠ Column order differs — reordering file2 to match file1...")
            df2 = df2[df1.columns]  # reorder columns
            df2.to_csv(file2, index=False)
            print(f"✅ Reordered file2 saved to: {file2}")

    # Convert to numpy arrays for calculation
    arr1 = df1.to_numpy(dtype=float)
    arr2 = df2.to_numpy(dtype=float)

    # Compute column means from arr1
    col_means = np.abs(np.mean(arr1, axis=0)) + eps  # shape: (num_columns,)


    # Compute element-wise diff
    #diff_arr = np.abs(arr1 - arr2) / (np.abs(arr1) + eps)
    
    # Compute element-wise diff relative to column mean
    diff_arr = np.abs(arr1 - arr2) / col_means  # broadcasting over columns
    diff_arr = np.round(diff_arr, 4)

    # Compute column-wise average diff (percentage)
    col_avg = np.round(np.mean(diff_arr, axis=0) * 100, 2)  # percentage

    # Prepare output DataFrame
    diff_df = pd.DataFrame(diff_arr, columns=df1.columns)
    avg_row = pd.DataFrame([col_avg], columns=df1.columns)
    avg_row.index = ["Average Diff (%)"]

    # Combine into one CSV: diffs + average row
    output_df = pd.concat([diff_df, avg_row])

    # Output file path
    output_file = os.path.join(os.path.dirname(file1), "csv_diff_output.csv")
    output_df.to_csv(output_file, index=False)

    print(f"✅ Comparison complete. Output saved to: {output_file}")
    print("Column-wise average percentage differences:")
    for col, avg in zip(df1.columns, col_avg):
        print(f"  {col}: {avg}%")

def main():
    parser = argparse.ArgumentParser(
        description="Compare two CSV files cell-wise and output percentage differences. "
                    "If column order differs, reorder file2 to match file1."
    )
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Small number to prevent division by zero (default: 1e-8)")
    parser.add_argument("--columns", type=str, default=None,
                        help="Comma-separated column names")
    
    args = parser.parse_args()

    try:
        compare_csv(args.file1, args.file2, args.columns, args.eps)
    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == "__main__":
    main()
                        
                        
                        
                        
                        
                        
                        
