import argparse
import os
import pandas as pd

def split_csv(csv_path: str, file1_name: str, file2_name: str, file2_columns: str, common_columns: str = None):
    """
    Split a CSV into two CSV files based on specified columns, with optional common columns.

    Parameters:
    - csv_path: Path to the input CSV file
    - file1_name: Name of the first output CSV file
    - file2_name: Name of the second output CSV file
    - file2_columns: Comma-separated list of column names for file2.csv
    - common_columns: Optional comma-separated list of columns to include in both files
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Parse file2 column list
    file2_col_list = [c.strip() for c in file2_columns.split(",") if c.strip()]

    # Validate file2 columns
    missing_cols_file2 = [c for c in file2_col_list if c not in df.columns]
    if missing_cols_file2:
        raise ValueError(f"Columns not found in CSV for file2: {missing_cols_file2}")

    # Parse common columns if provided
    common_col_list = []
    if common_columns:
        common_col_list = [c.strip() for c in common_columns.split(",") if c.strip()]
        missing_common = [c for c in common_col_list if c not in df.columns]
        if missing_common:
            raise ValueError(f"Common columns not found in CSV: {missing_common}")

    # Create DataFrames
    df2 = df[file2_col_list + common_col_list]
    df1 = df.drop(columns=file2_col_list)
    if common_col_list:
        # Ensure common columns are in df1
        for col in common_col_list:
            if col not in df1.columns:
                df1[col] = df[col]

    # Output paths
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    file1_path = os.path.join(base_dir, file1_name)
    file2_path = os.path.join(base_dir, file2_name)

    # Save CSVs
    df1.to_csv(file1_path, index=False)
    df2.to_csv(file2_path, index=False)

    print(f"✅ Saved {file1_path} with columns: {list(df1.columns)}")
    print(f"✅ Saved {file2_path} with columns: {list(df2.columns)}")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV into two CSV files based on specified columns, with optional common columns.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("file1_name", help="Name of the first output CSV file")
    parser.add_argument("file2_name", help="Name of the second output CSV file")
    parser.add_argument("file2_columns", help="Comma-separated list of column names for file2.csv")
    parser.add_argument("-c", "--common_columns", default=None, help="Optional comma-separated list of columns to include in both files")

    args = parser.parse_args()

    split_csv(
        args.csv_path,
        args.file1_name,
        args.file2_name,
        args.file2_columns,
        args.common_columns
    )

if __name__ == "__main__":
    main()
