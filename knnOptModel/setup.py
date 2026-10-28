from setuptools import setup, find_packages

setup(
    name="knnopt",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,  # important!
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tabulate",
        "joblib",
        "nlopt"
    ],
    entry_points={
        "console_scripts": [
            "knnopt-train=knnopt.cli_train:main",
            "knnopt-predict=knnopt.cli_predict:main",
            "knnopt-validate=knnopt.cli_validate:main",
            "combine-csv=knnopt.utils.combine_csv:main",
            "csv-diff=knnopt.utils.csv_diff:main",
            "data-process=knnopt.utils.data_processing:main",
            "dataset-cls=knnopt.utils.get_dataset_cls:main",
            "dataset-reg=knnopt.utils.get_dataset_reg:main",
            "correlation=knnopt.utils.pearson_coeffs:main",
            "split-csv=knnopt.utils.split_csv:main",
            "split-csv-rows=knnopt.utils.split_csv_by_row:main",
            "train-validate-split=knnopt.utils.train_validate_split:main",
        ]
    },
    author="Jianbo Huang",
    author_email="jianbo.huang22@gmail.com",  
    license="MIT",
    description="Optimized KNN Regressor/Classifier with NLopt weight optimization and CSV file process utilities",
    python_requires=">=3.10",
)
