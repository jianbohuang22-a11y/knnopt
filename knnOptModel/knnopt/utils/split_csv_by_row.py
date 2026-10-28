import argparse
import os
import pandas as pd

def split_csv_by_row(csv_path: str, file1_name: str, file2_name: str, file2_ratio: float):
    """
    Split a CSV file into two files by randomly selecting rows.

    Parameters:
    - csv_path: Path to the input CSV file (unchanged)
    - file1_name: Name of the first output CSV file (df1)
    - file2_name: Name of the second output CSV file (df2)
    - file2_ratio: Ratio of rows to put into file2.csv (0 < ratio < 1)
    """
    if not (0 < file2_ratio < 1):
        raise ValueError("file2_ratio must be between 0 and 1.")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Shuffle rows
    df_shuffled = df.sample(frac=1, random_state=None).reset_index(drop=True)

    # Calculate split index
    num_file2_rows = int(len(df_shuffled) * file2_ratio)

    # Split DataFrames
    df2 = df_shuffled.iloc[:num_file2_rows]
    df1 = df_shuffled.iloc[num_file2_rows:]

    # Output paths
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    file1_path = os.path.join(base_dir, file1_name)
    file2_path = os.path.join(base_dir, file2_name)

    # Save CSVs
    df1.to_csv(file1_path, index=False)
    df2.to_csv(file2_path, index=False)

    print(f"✅ Saved {file1_path} with {len(df1)} rows.")
    print(f"✅ Saved {file2_path} with {len(df2)} rows.")
    print(f"ℹ️ Original file {csv_path} remains unchanged.")

def main():
    parser = argparse.ArgumentParser(description="Split a CSV file into two files by randomly selecting rows.")
    parser.add_argument("csv_path", help="Path to the input CSV file (unchanged)")
    parser.add_argument("file1_name", help="Name of the first output CSV file")
    parser.add_argument("file2_name", help="Name of the second output CSV file")
    parser.add_argument("-r", "--file2-ratio", type=float, default=0.1, help="Ratio of rows to put into file2.csv (0 < ratio < 1)")

    args = parser.parse_args()

    try:
        split_csv_by_row(args.csv_path, args.file1_name, args.file2_name, args.file2_ratio)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
