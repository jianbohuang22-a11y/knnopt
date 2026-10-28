import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_wine,
    load_iris,
    fetch_california_housing,
    make_regression,
    make_classification
)
from knnopt.knnopt import KnnOpt

# Dataset registry: key → (loader function, model_type)
DATASET_LOADERS = {
    "breast_cancer": (load_breast_cancer, "cls"),
    "diabetes": (load_diabetes, "reg"),
    "wine": (load_wine, "cls"),
    "iris": (load_iris, "cls"),
    "california_housing": (fetch_california_housing, "reg"),
    "synthetic_regression": ("synthetic_regression", "reg"),
    "synthetic_classification": ("synthetic_classification", "cls")
}

def train_and_validate_dataset(dataset_name, case_dir,n_neighbors=5, test_size=0.2):
    """
    Train and validate a KnnOpt model using a specified dataset.
    
    Args:
        dataset_name (str): Dataset key from DATASET_LOADERS.
        case_dir (str): Directory to store model artifacts and CSV files.
        test_size (float): Fraction of remaining data (after validation split) to use for testing.
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader, model_type = DATASET_LOADERS[dataset_name]

    # === Load dataset ===
    if dataset_name == "synthetic_regression":
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    elif dataset_name == "synthetic_classification":
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        data = loader()
        X = data.data
        y = data.target
        feature_names = data.feature_names

    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    # === First split: reserve 10% for validation ===
    X_train, X_validate, y_train, y_validate = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_train0, y_train0, test_size=test_size, random_state=42
    )
    """
    # === Save training and validation CSVs ===
    os.makedirs(case_dir, exist_ok=True)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_validate, y_validate], axis=1)
    train_path = os.path.join(case_dir, "train_data.csv")
    val_path = os.path.join(case_dir, "input.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    # === Initialize KnnOpt ===
    knn = KnnOpt(
        features=list(feature_names),
        targets=["target"],
        model_type=model_type,
        n_neighbors=n_neighbors,
        weights="distance",
        metric="euclidean"
    )
    knn.case_dir = case_dir

    # === Train model ===
    knn.train(train_path)

    # === Validate model ===
    knn.validate(val_path)

    print(f"✅ Training and validation complete for dataset: {dataset_name}")


def example_breast_cancer():
    print('case name: breast_cancer -------------------')
    train_and_validate_dataset("breast_cancer", "case_breast_cancer", test_size=0.2)

def example_diabetes():
    print('case name: diabetes -------------------')
    train_and_validate_dataset("diabetes", "case_diabetes", test_size=0.25)

def example_wine():
    print('case name: wine -------------------')
    train_and_validate_dataset("wine", "case_wine", test_size=0.2)

def example_iris():
    print('case name: iris -------------------')
    train_and_validate_dataset("iris", "case_iris", test_size=0.2)

def example_california():
    print('case name:  california_housing -------------------')
    train_and_validate_dataset("california_housing", "case_california", n_neighbors=5, test_size=0.2)

def example_synthetic_regression():
    print('case name:  synthetic_regression -------------------')
    train_and_validate_dataset("synthetic_regression", "case_synth_reg", test_size=0.3)

def example_synthetic_classification():
    print('case name:  synthetic_classification -------------------')
    train_and_validate_dataset("synthetic_classification", "case_synth_cls", test_size=0.3)

# === Helper function to run cases ===
def run(case_name):
    # Mapping case names to example functions
    cases = {
        "breast_cancer": example_breast_cancer,
        "diabetes": example_diabetes,
        "wine": example_wine,
        "iris": example_iris,
        "california": example_california,
        "synthetic_regression": example_synthetic_regression,
        "synthetic_classification": example_synthetic_classification
    }

    if case_name == "all":
        for name, func in cases.items():
            print(f"\n🚀 Running case: {name}")
            func()
    elif case_name in cases:
        print(f"\n🚀 Running case: {case_name}")
        cases[case_name]()
    else:
        raise ValueError(f"❌ Unknown case name: {case_name}")

if __name__ == '__main__':
    run("all")
    
    
    
