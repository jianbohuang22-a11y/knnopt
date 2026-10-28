import argparse
import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification

def main():
    parser = argparse.ArgumentParser(
        description="Save shuffled dataset (built-in or synthetic) to CSV for validation UI."
    )
    
    # dataset_name as first positional argument
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=["iris", "wine", "breast_cancer", "synthetic"],
        help="Dataset to load: built-in or synthetic classification"
    )
    
    parser.add_argument(
        "-c", "--case-dir", type=str, default=os.getcwd(),
        help="Directory to save full_data.csv"
    )
    parser.add_argument(
        "-s", "--shuffle", action="store_true",
        help="Shuffle rows before saving"
    )
    parser.add_argument(
        "-r", "--record-number", type=int, default=500,
        help="Number of records for synthetic dataset"
    )
    parser.add_argument(
        "-n", "--noise-level", type=float, default=0.1,
        help="Noise level for synthetic dataset (0.0 to 1.0)"
    )

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
    elif args.dataset_name == "synthetic":
        X, y = make_classification(
            n_samples=args.record_number,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            flip_y=args.noise_level,
            random_state=42
        )
        
        df = pd.DataFrame(X.round(4), columns=[f"feature_{i}" for i in range(1, 11)])
        df["target"] = y

    # === Shuffle rows if requested ===
    if args.shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # === Save full dataset ===
    os.makedirs(args.case_dir, exist_ok=True)
    full_data_path = os.path.join(args.case_dir, "full_data.csv")
    df.to_csv(full_data_path, index=False)

    print(f"✅ Saved dataset ({args.dataset_name}) to {full_data_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"🔍 Columns: {list(df.columns)}")
    if args.shuffle:
        print("🔀 Data shuffled before saving.")
    if args.dataset_name == "synthetic":
        print(f"🛠 Synthetic dataset with {args.record_number} records and noise level {args.noise_level}")

if __name__ == "__main__":
    main()
