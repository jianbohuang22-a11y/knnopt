#!/usr/bin/env python3
import argparse
import pandas as pd
import os

def split_csv(input_csv, validate_size):
    # Read original CSV
    df = pd.read_csv(input_csv)

    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split index
    val_count = int(len(df_shuffled) * validate_size)

    # Split into validation and train
    df_val = df_shuffled.iloc[:val_count]
    df_train = df_shuffled.iloc[val_count:]

    # Prepare output paths
    base_dir = os.path.dirname(input_csv)
    train_path = os.path.join(base_dir, "train_data.csv")
    val_path = os.path.join(base_dir, "validation_input.csv")

    # Save files
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"✅ Train data saved to {train_path}")
    print(f"✅ Validation data saved to {val_path}")
    print(f"📄 Original file remains intact: {input_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="Split CSV into train and validation sets"
    )
    parser.add_argument("csv_path", help="Path to original CSV file")
    parser.add_argument("validate_size", nargs="?", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    args = parser.parse_args()

    split_csv(args.csv_path, args.validate_size)

if __name__ == "__main__":
    main()
