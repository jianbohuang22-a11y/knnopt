# knnopt/knnopt.py
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
import nlopt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score


class KnnOpt:
    def __init__(
        self, 
        features=[], 
        targets=[],
        model_type="reg", 
        n_neighbors=7,
        weights="distance", 
        metric="euclidean",
        optimizer="LN_COBYLA"
    ):
        """
        Initialize KnnOpt model configuration.

        Parameters:
            features (list): Feature column names.
            targets (list): Target column names.
            model_type (str): "reg" or "cls".
            n_neighbors (int): Number of neighbors.
            weights (str): KNN weights ("uniform" or "distance").
            metric (str): Distance metric.
            optimizer (str): NLopt optimizer name.
        """

        self.features = features
        self.targets = targets
        self.model_type = model_type
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.optimizer = optimizer
        self.n_features = len(self.features)
        self.n_targets = len(self.targets)
        self.scaler = None
        self.X_scaled = None
        self.y = None
        self.df = None
        self.W = np.ones((self.n_features, self.n_targets))
        self.models = {}
        self.case_dir = os.getcwd() 
        self.results = {}
        self.test_size = 0.3
        self.performance_rows = []
        self.upper_bounds = 20
       
        self.X_train = None 
        self.X_test = None 
        self.y_train = None 
        self.y_test = None
        self.mean_y_t = None
        self.t_idx = 0
        
        
    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=self.test_size, random_state=42
        )

        self.mean_y_t = np.mean(self.y, axis=0) if self.model_type == "reg" else None



    def load_train_data(self, csv_path: str, test_size: float=0.2):
        self.test_size = test_size
        
        df = pd.read_csv(csv_path)
        self.case_dir = os.path.dirname(os.path.abspath(csv_path))
       
        os.makedirs(self.case_dir, exist_ok=True)
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(df[self.features])
        self.y = df[self.targets].values
        self.data_split()
        self.df = df
        
            
    def objective(self, w, grad):
        X_train_w = self.X_train * w
        X_test_w = self.X_test * w
        y_train = self.y_train[:,self.t_idx]
        y_test = self.y_test[:,self.t_idx]
        mean_y_t = np.mean(y_test) 
        
        if self.model_type == "reg":
            model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)
            model.fit(X_train_w, y_train)
            y_pred = model.predict(X_test_w)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            return rmse / max(1, mean_y_t)
        else:
            model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)
            model.fit(X_train_w, y_train)
            y_pred_proba = model.predict_proba(X_test_w)
            loss = log_loss(y_test, y_pred_proba)
            return loss
            #y_pred = model.predict(X_test_w)
            #acc = accuracy_score(y_test, y_pred)
            #return 1 - acc  # minimize error

    def optimize_model(self):   
        t_idx = self.t_idx
        opt = nlopt.opt(getattr(nlopt, self.optimizer.upper(), nlopt.LN_COBYLA), self.n_features)
        opt.set_min_objective(self.objective)
        opt.set_lower_bounds(np.zeros(self.n_features))
        opt.set_upper_bounds(np.ones(self.n_features) * self.upper_bounds )
        opt.set_xtol_rel(1e-4)
        opt.set_maxeval(500)
        w_opt = opt.optimize(self.W[:, t_idx])
        self.W[:, t_idx] = w_opt
        return w_opt


    def correlation(self, t_idx: int):
      
        t_idx = self.t_idx
        target_name = self.targets[t_idx]
      
        if not hasattr(self, "df") or self.df is None:
            raise ValueError("DataFrame (self.df) is not loaded. Call load_train_data first.")

        # Compute correlations
        corr_list = []
        for idx_feature in range(len(self.features)):
            feature_name = self.features[idx_feature]
            # Pearson correlation between feature and target
            corr = self.df[feature_name].corr(self.df[target_name])
            corr_list.append(corr)
        
        return corr_list


    def finalize_regressor(self, t_idx, w_opt, target_name: str):
        X_train_w = self.X_train * w_opt
        X_test_w = self.X_test * w_opt
        y_train = self.y_train[:,t_idx]
        y_test = self.y_test[:,t_idx]
        
        final_model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        
        final_model.fit(X_train_w, y_train)

        # Predictions
        y_train_pred = final_model.predict(X_train_w)
        y_test_pred = final_model.predict(X_test_w)

        # Absolute MAE
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)


        # Determine normalization values
        mean_train = abs(np.mean(y_train))
        std_train = np.std(y_train, ddof=0)  # population stddev
        mean_test = abs(np.mean(y_test))
        std_test = np.std(y_test, ddof=0)

        denom_train = max(mean_train, std_train)
        denom_test = max(mean_test, std_test)

        norm_type_train = "mean(y_train)" if mean_train >= std_train else "rms(y_train)"
        norm_type_test = "mean(y_test)" if mean_test >= std_test else "rms(y_test)"

        # Relative MAE
        train_rel_mae = train_mae / denom_train
        test_rel_mae = test_mae / denom_test

        # Relative RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_rel_rmse = train_rmse / denom_train
        test_rel_rmse = test_rmse / denom_test


        # Append to performance rows
        quality_dict = {
            "Target": target_name,
            "train_rel_mae": round(train_rel_mae, 4),
            "test_rel_mae": round(test_rel_mae, 4),
            "train_rel_rmse": round(train_rel_rmse, 4),
            "test_rel_rmse": round(test_rel_rmse, 4),
            "Train_error_norm_by": norm_type_train,
            "Test_error_norm_by": norm_type_test
        }
        self.performance_rows.append(quality_dict)

        self.results[target_name] = {
                "train_rel_mae": float(train_rel_mae),
                "test_rel_mae": float(test_rel_mae),
                "train_rel_rmse": float(train_rel_rmse),
                "test_rel_rmse": float(test_rel_rmse),
                "Train_error_norm_by": norm_type_train,
                "Test_error_norm_by": norm_type_test
            }
        
        self.models[target_name] = final_model
        

    def finalize_classifier(self, t_idx, w_opt, target_name: str):
        X_train_w = self.X_train * w_opt
        X_test_w = self.X_test * w_opt
        y_train = self.y_train[:,t_idx]
        y_test = self.y_test[:,t_idx]
        
        final_model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
           
        final_model.fit(X_train_w, y_train)

        # Predictions for train and test
        y_pred_train = final_model.predict(X_train_w)
        y_pred_test = final_model.predict(X_test_w)

        # Accuracy
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        # Confusion matrix (on test set)
        cm = confusion_matrix(y_test, y_pred_test).tolist()  # convert to list for JSON serialization
        quality_dict = {
            "Target": target_name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "confusion_matrix": cm
        }

        # Append to performance rows
         
        self.performance_rows.append(quality_dict)

        self.results[target_name] = quality_dict
        self.models[target_name] = final_model

    def train(self, csv_path: str, test_size: float=0.2):
        self.case_dir = os.path.dirname(os.path.abspath(csv_path))
        self.test_size = test_size
        try:
            self.load_train_data(csv_path, test_size)
        except Exception as e:
            message = f'error load train data in KnnOpt train_model: {e}'
            print(message)
            return None
            
        for t_idx, target_name in enumerate(self.targets):
            self.t_idx = t_idx
            w_opt = self.optimize_model()
            
            if self.model_type == "reg":
                self.finalize_regressor(t_idx, w_opt, target_name)
            else:
                self.finalize_classifier(t_idx, w_opt, target_name)

        # Save artifacts
        joblib.dump(self.scaler, os.path.join(self.case_dir, "scaler.pkl"))
        np.save(os.path.join(self.case_dir, "weights.npy"), self.W)
        joblib.dump(self.models, os.path.join(self.case_dir, "knn_models.pkl"))

        # Save performance CSV
        print('performance_rows:', self.performance_rows)
    
        perf_df = pd.DataFrame(self.performance_rows)
        perf_df.to_csv(os.path.join(self.case_dir, "performance.csv"), index=False, encoding="utf-8")

        print("\n✅ Training complete!")
        print("\nOptimized Weights Matrix:")
       
        print("\nPerformance Metrics:")
    
        print(tabulate(perf_df, headers="keys", tablefmt="pretty"))
        
        return self.results
    
    
    def predict(self, input_csv=None):
        """
        Predict using the trained model.
        Recomputes scaling based on training data + new input data.
        """

        # === Paths ===
        train_data_path = os.path.join(self.case_dir, "train_data.csv")
        weights_path = os.path.join(self.case_dir, "weights.npy")
        models_path = os.path.join(self.case_dir, "knn_models.pkl")

        if not all(os.path.exists(p) for p in [train_data_path, weights_path, models_path]):
            raise FileNotFoundError("❌ Missing required artifacts in case_dir.")

        # === Load weights & models ===
        self.W = np.load(weights_path)
        self.models = joblib.load(models_path)

        # === Load input CSV ===
        if input_csv is None:
            input_csv = os.path.join(self.case_dir, "input.csv")
        df_input = pd.read_csv(input_csv)

        # === Load training data ===
        df_train = pd.read_csv(train_data_path)

        # Check for missing features
        missing_features = [f for f in self.features if f not in df_input.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        # === Combine train + input for scaling ===
        combined_df = pd.concat([df_train[self.features], df_input[self.features]], ignore_index=True)
    
        # === Refit scaler ===
        scaler = MinMaxScaler()
        scaler.fit(combined_df)

        # === Scale input features ===
        X_scaled = scaler.transform(df_input[self.features])

        # === Predict per target ===
        predictions = {}
        for idx, target_name in enumerate(self.models.keys()):
            X_weighted = X_scaled * self.W[:, idx]
            model = self.models[target_name]
            y_pred = model.predict(X_weighted)
            predictions[target_name] = y_pred

        # === Build output DataFrames ===
        predictions_df = pd.DataFrame(predictions)
        predictions_full_df = pd.concat([df_input.reset_index(drop=True), predictions_df], axis=1)

        return predictions_df, predictions_full_df


    def validate(self, input_csv=None):
        """
        Validate predictions against actual target values in input CSV.
        Saves validation results to CSV files.
        """
        # Get predictions
        pred_df, _ = self.predict(input_csv)

        # Load input CSV
        if input_csv is None:
            input_csv = os.path.join(self.case_dir, "input.csv")
        df_input = pd.read_csv(input_csv)

        # Ensure targets exist
        missing_targets = [t for t in self.targets if t not in df_input.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        # Choose validation method
        if self.model_type == "reg":
            self._validate_regression(df_input, pred_df)
        else:
            self._validate_classification(df_input, pred_df)
            
    
    def _validate_regression(self, df_input, pred_df):
        """
        Regression validation: relative MAE and RMSE normalized by max(abs(mean), stddev).
        """
        results_data = []
        for target in self.targets:
            y_true = df_input[target].values
            y_pred = pred_df[target].values

            mean_true = abs(np.mean(y_true))
            std_true = np.std(y_true, ddof=0)
            denom = max(mean_true, std_true)
            norm_type = "mean" if mean_true >= std_true else "rms"

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rel_mae = mae / denom
            rel_rmse = rmse / denom

            results_data.append({
                "Target": target,
                "Mean": mean_true,
                "StdDev": std_true,
                "Normalized_by": norm_type,
                "RelMAE": round(rel_mae, 4),
                "RelRMSE": round(rel_rmse, 4)
            })

        # Column-average relative MAE
        avg_rel_mae = np.mean([r["RelMAE"] for r in results_data])
        results_data.append({
            "Target": "Average",
            "Mean": np.nan,
            "StdDev": np.nan,
            "Normalized_by": "",
            "RelMAE": round(avg_rel_mae, 4),
            "RelRMSE": np.nan
        })

        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.case_dir, "regression_validation.csv")
        results_df.to_csv(results_path, index=False, encoding="utf-8")
        print(f"✅ Regression validation results saved to '{results_path}'")
 
        
    def _validate_classification(self, df_input, pred_df):
        """
        Classification validation: element-wise correctness and column-average accuracy.
        """
        # Actual and predicted arrays
        actual_arr = df_input[self.targets].to_numpy(dtype=int)
        pred_arr = pred_df[self.targets].to_numpy(dtype=int)

        # Element-wise correctness (1 = correct, 0 = incorrect)
        match_arr = (actual_arr == pred_arr).astype(int)

        # Column-average accuracy in %
        col_avg_acc = np.round(np.mean(match_arr, axis=0) * 100, 2)

        # === Save element-wise correctness ===
        correct_df = pd.DataFrame(match_arr, columns=self.targets)
        caption_df = pd.DataFrame(
            [["element-wise correct predictions (1=true, 0=false)"] + [""] * (len(self.targets) - 1)],
            columns=self.targets
        )
        correct_df = pd.concat([caption_df, correct_df], ignore_index=True)
        correct_path = os.path.join(self.case_dir, "correct_prediction.csv")
        correct_df.to_csv(correct_path, index=False, encoding="utf-8")

        # === Save column-average accuracy ===
        summary_df = pd.DataFrame([col_avg_acc], columns=self.targets)
        caption_summary_df = pd.DataFrame(
            [["column-averaged prediction accuracy (%)"] + [""] * (len(self.targets) - 1)],
            columns=self.targets
        )
        summary_df = pd.concat([caption_summary_df, summary_df], ignore_index=True)
        summary_path = os.path.join(self.case_dir, "accuracy_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")

        print(f"✅ Classification validation complete.")
        print(f"   Element-wise correctness saved to '{correct_path}'")
        print(f"   Column-average accuracy saved to '{summary_path}'")

     
