#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    load_diabetes, fetch_california_housing, make_regression
)

def cli_save_full_data():
    parser = argparse.ArgumentParser(description="Save shuffled full dataset to CSV for validation UI.")
    parser.add_argument("dataset_name", type=str,
                        choices=[
                            "iris", "wine", "breast_cancer",
                            "diabetes", "california_housing", "synthetic_regression"
                        ],
                        help="Built-in dataset to load")
    parser.add_argument("case_dir", nargs="?", default=os.getcwd(),
                        help="Directory to save full_data.csv (default: current directory)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle rows before saving")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples for synthetic regression (only for synthetic_regression)")
    parser.add_argument("--n_features", type=int, default=5,
                        help="Number of features for synthetic regression (only for synthetic_regression)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for synthetic regression (only for synthetic_regression)")
    args = parser.parse_args()

    # === Load dataset ===
    if args.dataset_name == "iris":
        data = load_iris(as_frame=True)
        df = data.frame
    elif args.dataset_name == "wine":
        data = load_wine(as_frame=True)
        df = data.frame
    elif args.dataset_name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    elif args.dataset_name == "diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame
    elif args.dataset_name == "california_housing":
        data = fetch_california_housing(as_frame=True)
        df = pd.concat([data.data, pd.Series(data.target, name="target")], axis=1)
    elif args.dataset_name == "synthetic_regression":
        X, y = make_regression(
            n_samples=args.n_samples,
            n_features=args.n_features,
            noise=args.noise,
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(args.n_features)])
        df["target"] = y

    # === Shuffle rows if requested ===
    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    # === Round floats to 4 decimal places ===
    df = df.round(4)


    # === Save full dataset ===
    os.makedirs(args.case_dir, exist_ok=True)
    full_data_path = os.path.join(args.case_dir, "full_data.csv")
    df.to_csv(full_data_path, index=False)

    print(f"✅ Saved full dataset ({args.dataset_name}) to {full_data_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"🔍 Columns: {list(df.columns)}")
    if args.shuffle:
        print("🔀 Data shuffled before saving.")

if __name__ == "__main__":
    cli_save_full_data()
