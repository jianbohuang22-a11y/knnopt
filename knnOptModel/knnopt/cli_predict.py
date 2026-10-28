# knnopt/cli_predict.py
import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="Predict using trained KNN Optimizer (Regression or Classification)"
    )
    parser.add_argument(
        "model_type",
        nargs="?",
        default="reg",
        choices=["reg", "cls"],
        help="Model type: reg=regression (default), cls=classification"
    )
    parser.add_argument(
        "model_dir_or_file",
        nargs="?",
        default=".",
        help="Model directory or full input CSV path (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        default="predictions.csv",
        help="Output CSV file name (default: predictions.csv)"
    )

    args = parser.parse_args()

    # Determine model_dir and input_csv
    if args.model_dir_or_file.lower().endswith(".csv"):
        input_csv = os.path.abspath(args.model_dir_or_file)
        model_dir = os.path.dirname(input_csv)
    else:
        model_dir = os.path.abspath(args.model_dir_or_file)
        input_csv = os.path.join(model_dir, "input.csv")

    # Validate paths
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory '{model_dir}' not found.")
        sys.exit(1)
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input CSV '{input_csv}' not found.")
        sys.exit(1)

    # Load artifacts
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    weights_path = os.path.join(model_dir, "weights.npy")
    models_path = os.path.join(model_dir, "knn_models.pkl")

    if not all(os.path.exists(p) for p in [scaler_path, weights_path, models_path]):
        print("❌ Error: Missing model artifacts in model_dir.")
        sys.exit(1)

    scaler = joblib.load(scaler_path)
    W = np.load(weights_path)
    models = joblib.load(models_path)

    # Load input data
    df = pd.read_csv(input_csv)

    # Ensure all feature columns exist
    features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else list(df.columns)
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"❌ Error: Missing feature columns in input CSV: {missing_features}")
        sys.exit(1)

    # Scale features
    X_scaled = scaler.transform(df[features])

    # Predict for each target/class
    predictions = {}
    for idx, target_name in enumerate(models.keys()):
        X_weighted = X_scaled * W[:, idx]
        y_pred = models[target_name].predict(X_weighted)

        if args.model_type == "reg":
            predictions[target_name] = np.round(y_pred, 4)
        else:
            predictions[target_name] = y_pred  # classification labels

    # Save predictions.csv (just predictions)
    pred_df = pd.DataFrame(predictions)
    pred_csv_path = os.path.join(model_dir, args.output)
    pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8")

    # Save predictions_full.csv (input + predictions)
    pred_full_df = df.copy()
    for col, vals in predictions.items():
        pred_full_df[col + "_pred"] = vals
    pred_full_csv_path = os.path.join(model_dir, "predictions_full.csv")
    pred_full_df.to_csv(pred_full_csv_path, index=False, encoding="utf-8")

    print(f"✅ Predictions saved to {pred_csv_path}")
    print(f"✅ Full predictions saved to {pred_full_csv_path}")
    print(pred_df)

if __name__ == "__main__":
    main()
