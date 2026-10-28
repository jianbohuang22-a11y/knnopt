import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="Validate trained KNN Optimizer model with relative difference (%)"
    )
    parser.add_argument(
        "model_type",
        choices=["reg", "cls"],
        default="reg",
        help="Model type: reg=regression, cls=classification"
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing trained model artifacts"
    )
    parser.add_argument(
        "input_csv",
        help="CSV file containing features and target columns"
    )
    parser.add_argument(
        "-o", "--output",
        default="validation_results.csv",
        help="Output CSV file name for results (default: validation_results.csv)"
    )

    args = parser.parse_args()

    # Validate paths
    model_dir = os.path.abspath(args.model_dir)
    input_csv = os.path.abspath(args.input_csv)

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

    # Compute relative difference for regression
    results_data = []
    if args.model_type == "reg":
        for target_name in models.keys():
            if target_name not in df.columns:
                print(f"Error: Target column '{target_name}' not found in input CSV.")
                sys.exit(1)

            y_true = df[target_name].values
            y_pred = predictions[target_name]

            # Mean and StdDev of true values
            mean_true = np.mean(y_true)
            std_true = np.std(y_true, ddof=0)  # population stddev

            # Average absolute difference
            avg_diff = np.mean(np.abs(y_true - y_pred))

            # Normalization choice: max(abs(mean), stddev)
            norm_value = max(abs(mean_true), std_true)
            norm_type = "mean" if abs(mean_true) >= std_true else "stddev"

            # Relative difference in percentage
            rel_diff_pct = (avg_diff / norm_value) * 100

            results_data.append({
                "Target": target_name,
                "Mean": round(mean_true,4),
                "StdDev": round(std_true),
                "NormType": norm_type,
                "NormValue": round(norm_value,4),
                "AvgDiff": round(avg_diff,4),
                "RelDiff%": round(rel_diff_pct,4)
            })

        # Column-average of relative difference
        avg_rel_diff_pct = np.mean([r["RelDiff%"] for r in results_data])
        results_data.append({
            "Target": "Average",
            "Mean": np.nan,
            "StdDev": np.nan,
            "NormType": "",
            "NormValue": np.nan,
            "AvgDiff": np.nan,
            "RelDiff%": avg_rel_diff_pct
        })

        # Save results to CSV
        results_df = pd.DataFrame(results_data)
        results_csv_path = os.path.join(model_dir, args.output)
        results_df.to_csv(results_csv_path, index=False, encoding="utf-8")

        print(f"✅ Validation complete. Relative difference results saved to '{results_csv_path}'.")

    else:
        # Ensure all feature columns exist
        features = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else list(df.columns)
        missing_features = [f for f in features if f not in df.columns]
        missing_targets = [t for t in models.keys() if t not in df.columns]
        if missing_features or missing_targets:
            print(f"❌ Error: Missing columns: {missing_features + missing_targets}")
            sys.exit(1)

        # Prepare features
        X_scaled = scaler.transform(df[features])

        # Predict for each target/class
        predictions = {}
        for idx, target_name in enumerate(models.keys()):
            X_weighted = X_scaled * W[:, idx]
            y_pred = models[target_name].predict(X_weighted)
            predictions[target_name] = y_pred.astype(int)

        # Actual values
        actual_arr = df[list(models.keys())].to_numpy(dtype=int)
        pred_arr = pd.DataFrame(predictions)[list(models.keys())].to_numpy(dtype=int)

        # Element-wise correctness (1 = correct, 0 = incorrect)
        match_arr = (actual_arr == pred_arr).astype(int)

        # Column-average accuracy in %
        col_avg_acc = np.round(np.mean(match_arr, axis=0) * 100, 2)

        # === Save element-wise correctness ===
        accuracy_df = pd.DataFrame(match_arr, columns=list(models.keys()))
        accuracy_csv_path = os.path.join(model_dir, "correct_prediction.csv")
        # Insert caption row at top
        caption_df = pd.DataFrame(
            [["element-wise correct predictions (1=true, 0=false)"] + [""] * (len(models.keys()) - 1)],
            columns=list(models.keys())
        )
        accuracy_df = pd.concat([caption_df, accuracy_df], ignore_index=True)
        accuracy_df.to_csv(accuracy_csv_path, index=False, encoding="utf-8")

        # === Save column-average accuracy ===
        summary_df = pd.DataFrame([col_avg_acc], columns=list(models.keys()))
        accuracy_summary_path = os.path.join(model_dir, "accuracy_summary.csv")
        # Insert caption row at top
        caption_summary_df = pd.DataFrame(
            [["column-averaged prediction accuracy (%)"] + [""] * (len(models.keys()) - 1)],
            columns=list(models.keys())
        )
        summary_df = pd.concat([caption_summary_df, summary_df], ignore_index=True)
        summary_df.to_csv(accuracy_summary_path, index=False, encoding="utf-8")

        print(f"✅ Classification validation complete.")
        print(f"   Element-wise correctness saved to '{accuracy_csv_path}'")
        print(f"   Column-average accuracy saved to '{accuracy_summary_path}'")


    # Save predictions_full.csv (input + predictions)
    pred_full_df = df.copy()
    for col, vals in predictions.items():
        pred_full_df[col + "_pred"] = vals
    pred_full_csv_path = os.path.join(model_dir, "predictions_full.csv")
    pred_full_df.to_csv(pred_full_csv_path, index=False, encoding="utf-8")

    print(f"✅ Predictions saved to '{pred_full_csv_path}'.")


if __name__ == "__main__":
    main()
