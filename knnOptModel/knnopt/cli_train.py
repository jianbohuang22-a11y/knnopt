import argparse
import os
import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
from knnopt.knnopt import KnnOpt

def main():
    parser = argparse.ArgumentParser(description="Train KNN Optimizer (Regression or Classification)")
    parser.add_argument("csv", help="Path to CSV dataset file")
    parser.add_argument("-f", "--features", type=str, required=True,
                        help="Comma-separated feature column names")
    parser.add_argument("-t", "--targets", type=str, required=True,
                        help="Comma-separated target column names")
    parser.add_argument("--model-type", choices=["reg", "cls"], default="reg")
    parser.add_argument("-n", "--n_neighbors", type=int, default=5)
    parser.add_argument("-w", "--weights", choices=["uniform", "distance"], default="distance")
    parser.add_argument("-m", "--metric", choices=["euclidean", "manhattan"], default="euclidean")
    parser.add_argument("--optimizer", default="LN_COBYLA")
    # ✅ Add test_size argument
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="Fraction of dataset to use for testing (default: 0.3)")
    args = parser.parse_args()

    # Validate CSV file
    if not args.csv.lower().endswith(".csv"):
        print("Error: Must be a CSV file.")
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"Error: File '{args.csv}' not found.")
        sys.exit(1)

    # Split features and targets into lists
    features_list = [f.strip() for f in args.features.split(",") if f.strip()]
    targets_list = [t.strip() for t in args.targets.split(",") if t.strip()]
    print('features_list:', features_list)
    
    # Load CSV
    df = pd.read_csv(args.csv)

    # Validate columns
    for col in features_list + targets_list:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV.")
            sys.exit(1)

    # Train model
    model = KnnOpt(
        features=features_list,
        targets=targets_list,
        model_type=args.model_type,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric,
        optimizer=args.optimizer
    )

    result = model.train(args.csv, test_size=args.test_size)



if __name__ == "__main__":
    main()
