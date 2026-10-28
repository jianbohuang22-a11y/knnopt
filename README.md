# knnopt
Optimized KNN Regressor/Classifier with NLopt weight optimization. The package also contains CSV utilities for data alalytics and machine-learning
# knnopt

**Optimized KNN Regressor/Classifier with NLopt weight optimization**

`knnopt` is a Python package that extends the standard K-Nearest Neighbors (KNN) algorithm by automatically optimizing **feature-target weights** 
using the [NLopt](https://nlopt.readthedocs.io/en/latest/) library.  
This allows KNN models to adapt feature importance for each target variable, improving accuracy and robustness.
Test and validation cases show that it has better accuracy than other commonly-used machine-learning
models for most of the test cases, while on other cases, its accuracy has similar level
with other models. Our validation cases show that, it does not have over-fitting issues. Test and
validation accuracy is consistent across different cases, and performs well for cases
with big sample records such as 10000 records, as well as cases with small number of records
around 100. 

This package also contain a number of utilities for handling CSV data file. They are designed to help
with general data science and machine learning tasks.

---

## 🚀 Features

- **Supports Regression and Classification**
- **Automatic Feature Weight Optimization** per target using NLopt
- **Scales Features** with `MinMaxScaler` for consistent distance metrics
- **Saves Models and Scalers** for later prediction
- **Performance Tracking** via `performance.csv`
- **Customizable KNN Parameters** (`n_neighbors`, `weights`, `metric`)
- **Choice of NLopt Optimizers** (e.g., `LN_COBYLA`, `LD_MMA`, etc.)
- **CSV utilities** a set of utilities for handling CSV data, useful for data analysis and machine learning tasks

---

## 🛠 Technology Overview

This model is implemented as a Python class KnnOpt. It consists of methods for loading training data,
scale dataset, optimization objective function, optimization model, feature-target correlation,
train model, performance metrics calculation, targets prediction with new, untrained feature records,
model validation with untrained dataset which contains both features and targets columns.

Apart from the class implementation, this package also contains three CLI utility files for model training, 
prediction and validation, and a set of utilities for CSV file handling and training data preperation.
The knnopt-model-related utilities are:

```
cli_predict.py  cli_train.py  cli_validate.py 
```
These utilities allow us to train, validate KnnOpt model, and use the trained model for targets
prediction using commands run on terminals. This approach simplifies the usage of machine-learning
models, especially helpful for researchers/engineers who do not have much experiences in 
data science and computer languages. 

This package also includes utilites for preparing CSV data files prior to train a model.
They can be used for any data analysis and machine-learning jobs involving CSV data. The utility files are:


```combine_csv.py  csv_diff.py  data_processing.py  get_dataset_cls.py  
   get_dataset.py  pearson_coeffs.py  split_csv_by_row.py  split_csv.py  
   train_validate_split.py
```

## KnnOpt Model creation

The syntax to create a KnnOpt model is:

```python
  model = KnnOpt(
        features=['f0','f1','f2'...], # feature-column names list
        targets=['t0','t1','t2'...], # target-column names list
        model_type='reg', # cls
        n_neighbors=5, #any suitable int >= 1, 3 to 7 are suitable choices
        weights='distance', #uniform
        metric='euclidean', #manhattan
        optimizer='LN_COBYLA' #LN_BOBYQA, GN_DIRECT
    )
```
The most influential parameter is: n_neighbors. The weights, metric and optimizer default values are suitable for majority cases.

## Train KnnOpt Model

Once KnnOpt model instance is created, the next step is to train the model. The syntax is:

```python
model.train(csv_path, test_size=0.3)

```
where csv_path is the absolute path to the training data file, test_size is fraction of data used for model testing.
Note that:

- All the model-related data files, results files, weights, pipeline file will be saved in the same directory as the training data file.
- test_size has a substantial effect on model accuracy. A value range of [0.2, 0.35] is a suitable choice.
- The train method will automatically call data pre-processing method to scale the data
- All data columns in the training data file needs to contain numerical data only
- If your CSV data file contain categorical data, and/or missing data, you can use the data_processing utility to pre-process the data file first.
- The test accuracy or test error metrics are the real measurement of model accuracy. High train accuracy does not mean the model
is accurate, because the target values for evaluation are used in training the model.
- In evaluating the test accuracy/error, the target values are not used for model training, so it is a real representation of the
model performance.
- During training, the weights of feature(i) to target(j), W(i,j) is optimized in such a way that the test error becomes minimized.
The optimization procedure is automatically called, and the final weights array W is used to fit the model.
- The feature-target weight can be seen as the importance factor of a feature influencing a target, therefore, optimized weights will
help KNN model to find the best match between a new record and the neighbouring records, thus increase prediction accuracy. 
- The training process will create and store 4 files in the case directory:
```
 knn_models.pkl  scaler.pkl   weights.npy   performance.csv 
```

## Workflow of the model training process

1. Load CSV containing features and targets.
1. Scale Features using MinMaxScaler.
1. Split Data into train/test sets.
1. Optimize Feature Weights for each target using NLopt:
1. Minimizes RMSE for regression. Maximizes accuracy/F1 for classification.
1. Train KNN Models with optimized weights.
1. Save Outputs in case directory:
- **scaler.pkl** — feature scaler
- **weights.npy** — optimized weights
- **knn_models.pkl** — trained KNN model
- **performance.csv** — metrics for each target


## Validate KnnOpt Model

The test accuracy is a good indication of machine-learning model performance, but it does not ensure the prediction will
have the same accuracy due to potential parameter overfitting issue. To check whether a KnnOpt model is overfitted, a validation method is created. 
The idea is that we keep a fraction of the sample data which contains both features and targets out of the train-test loop, therefore it is
true representation of model prediction except that we know the true targets. To validate KnnOpt model against a given dataset, the procedure is:

- Split the whole data into training data and validation data, using Scikit-learn train-test split function. You can control the fraction of the data
to be used for validation and randomly mix up the data for a fair validation.
- The remaining data  train data can then be fed into a KnnOpt model for training.
- This process can be done by using this packages utility code: **train_validate_split.py**
- The process will create two files in the case directory, while the original data file is unchanged:
```
train_data.csv    validation_input.csv    
```
- To validate the trained model, you can either use the validation utility in cli_validate.py or use the class method
model.validate(input_csv), where input_csv is the filename which contains validation input data. The file must already be in the
case directory

## KnnOpt class methods

The KnnOpt class contains the following methods:

- **load_train_data**: load training dataset in given CSV file. The data file contains features columns, targets columns, and can contain
   other columns not used for training/prediction. It also scale the data using Scikit-lean MinMaxScaler() function
- **data_split**: split the loaded data into train and test parts
- **objective**: objective function to minimise the test error
- **optimize_model**: Procedure to get the optimized weights
- **correlation**: helper method to get correlation coefficients of all features with respect to targets[t_idx]
- **finalize_regressor**: calculate and save the performance metrics for regressor
- **finalize_classifier**: calculate and save the performance metrics for classifier
- **train**: scale training data, train the model, optimize weights, save model file, scalar file and weights file in case directory
- **predict**: use the trained model to predict targets based on unseen features data
- **validate**: use untrained input file which contains both features and targets and the trained model to predict
targets, find the difference between true values and predicted values, output the relative errors, prediction
accuracy to case directory. This is an effective tool to check whether the model overfits the training data.

## Utilities

1. Train KnnOpt model **cli_train.py**: CLI command to train a KnnOpt model. 

- Usage example for regressor

Under a terminal, type the following command:
```
cd ~/knnOptModel/knnopt/examples/metal_strength
knn-train train_data.csv -f f0,f1,f2,f3,f4 -t target1,target2,target3,target4 -n 7 --test-size 0.2
```
You can also put the command in a train.sh file, make it executable, go to the directory, then under terminal window,
type:
```
train.sh
```
Note that model_type default is 'reg', therefore for regression application, the --model-type option is not needed.

- Usage example for classifier

Under a terminal type the following command:

```
knnopt-train train_data.csv -f f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11 \
-t medical_event --model-type cls  -n  7 --test-size 0.25
```

Note that the 

```
--model-type cls 
```
is needed, as classifier is not the default model type. Put the command in a train.sh file is a better choice for reuse
and for keeping record.

2. Validate KnnOpt model **cli_validate.py**: CLI command to validate a trained KnnOpt model. 
- Usage example for regressor: the bash file validate.sh for command is as follows

```
knnopt-validate reg ./ validate_input.csv
```
where ./ is the directory containing the validation input file. You can use absolute path name
and any filename (must be with extension .csv) for the validation input file.

- Usage example for classifier: the bash file validate.sh for command is as follows

```
knnopt-validate cls ./ validate_input.csv
```
where ./ is the directory containing the validation input file. You can use absolute path name
and any filename (must be with extension .csv) for the validation input file.

3. KnnOpt model prediction **cli_predict.py**: CLI command to predict the targets of giving features data, by using a trained KnnOpt model. 
- Usage example for regressor: the bash file predict.sh for command is as follows


```
knnopt-predict reg input.csv
```

where reg is the model_type, input.csv is input data path


- Usage example for classifier: the bash file predict.sh for command is as follows


```
knnopt-predict cls input.csv
```

where cls is the model_type, input.csv is input data path

4. combine-csv **utils/combine_csv.py**: combine two CSV files the same number of rows but
different column names. 

- Usage:
```
combine-csv file1.csv file2.csv
```
- Both files must contain a header and have the same number of rows
- Combined CSV data will be saved as combined_csv.csv, and in the same directory as file1.csv
- Duplicated column names in file2.csv will be ignored
- This utility is useful to combine feature data and target data, if they are originally not in the same file
- It is useful for any data analytics, not specific to KnnOpt model


5. csv-diff **utile/csv_diff.py**: find data difference between two CSV files. 

- Usage example:

```
csv-diff file1 file2 --columns f1,f2,f3,f4 --eps 1.0e-4
```

where file1 and file2 are the paths of the two CSV files, columns is a string containing comma-separated column names to find the differences with.
The two files may contain other columns, but names in **--columns** must be exist in both files. Both files must have the same number of rows.
eps is a small number to prevent division-by-zero. The cell-wise absolute difference is normalized by the mean of the associated column.
If **--columns** is missing, all columns will be compared.


6. data-process **utile/data_processing.py**: CSV preprocessing utility: auto missing data handling, categorical/boolean to numeric, float rounding 

- Usage example:

```
data-process input_csv -s skip-columns -e extra-category -m missing-value
```

where input_csv is the file path to the csv file to be processed. All processed data will be written in the same directory as the input_csv.
skip-columns is a string of comma-separated names, those columns will not be handled. The utility will automatically identify category data columns,
However, if some data columns seemingly numerical such as postcodes, but actually category, you can use the -e option to define them.
The -s option missing-value is for defining missing value cells in the dataset, such as NA, N/A, nan, or empty string, default is empty string("").
Processed data will be saved into file processed.csv.

7. dataset-cls **utile/get_dataset_cls.py**: get CSV data from Scikit-learn  classification datasets and save as CSV files. 

- Usage example:

```
dataset-cls dataset_name -c case_directory -s -r records -n noise
```

where: dataset_name includes iris, wine, breast_cancer, synthetic. -s means the data rows are randomly shuffled for model training.
case_directory default to current working directory. records(default 500) and noise(default 0.1) are only for synthetic
dataset.

- Output data file: output data will be saved as full_data.csv, in case_directory



8. dataset-reg **utile/get_dataset_reg.py**: get CSV data from Scikit-learn  regression datasets and save as CSV files. 

- Usage example:

```
dataset-reg dataset_name -c case_directory -f nfeatures -s -r records -n noise
```

where: dataset_name includes wine, california_housing, synthetic. -s means the data rows are randomly shuffled for model training.
case_directory default to current working directory. nfeatures(default 10), records(default 1000) and noise(default 0.1) are only for synthetic
dataset.  

- Output data file: output data will be saved as full_data.csv, in case_directory


9. correlation **utile/pearson_coeffs.py**: calculates correlation coefficients of each feature column to a target column. 

- Usage example:

```
correlation csv_path targets -s skip_columns
```

where: csv_path is the data file to be handled, targets is a comma-separated string containing the target names. skip_columns
is a comma-separated string containing names of columns in the data file to be skipped for correlation.

- Output data file: output data files are: correlation_coefficients.csv and normalized_coefficients.csv. They will be saved in the same directory as csv_path file.


10. split-csv **utile/split_csv.py**: split a CSV file into two CSV files based on specified columns.

- Usage example:
```
split-csv csv_path file1_name file2_name file2_columns -c common_columns
```

where:
- **csv_path** : path to the input CSV file
- **file1_name**: name of the first output CSV file
- **file2_name**: name of the second output CSV file
- **file2_columns** : comma-separated string containing column names for file2.csv
- **common_columns**: Optional comma-separated list of columns to include in both files

This utility is useful for spliting a file into feature file and target file.


11. split-csv-rows **utile/split_csv_by_row.py**: split a CSV file into two files by randomly selecting rows.

- Usage example:
```
split-csv-rows csv_path file1_name file2_name -r file2_ratio
```

where:
- **csv_path** : path to the input CSV file
- **file1_name**: name of the first output CSV file
- **file2_name**: name of the second output CSV file
- **file2_ratio**: ratio of rows to put into file2.csv (0 < ratio < 1, default 0.1)

This utility is useful for spliting a file into train file and validation file.


12. train-validate-split **utile/train_validate_split.py**: split a CSV file into train file and validate file.

- Usage example:
```
train-validate-split csv_path -r validate_size
```

where:
- **csv_path** : path to the input CSV file

- **validate_size**: fraction of data for validation (0 < ratio < 1, default 0.1)

- The splitted data will be saved into train_data.csv and validation_input.csv, stored in the same
directory as the csv_path.

This utility is useful for spliting a file into train file and validation file.


## Examples

A number of examples are provided in the examples/ folder. There are two types of examples:
1. Examples using datasets provided by scikit-learn, for create and run those examples, do:
```
cd knnOptModel/knnopt/examples
python3 knnopt_validate.py
```

There are 7 cases in this category: 4 classification cases:  breast_cancer, wine, iris, synthetic_classification and 3
regression cases: diabetes, california_housing, synthetic_regression

2. Datasets provided by this package

The package includes 5 additional test and validation cases: 3 regression cases: metal_strength, heat_tr_coef, matprop,
2 classification cases: medical_event, claim_predict.

To run those cases, cd to the a case, then type ./run.sh. You may change the parameters in the run.sh file
to see the effect of each parameter.



## Installation

To install this package, you can download or clone the source package. Assuming the package is stored
in the $HOME directory, then under terminal, do:

```
cd ~/knnOptModel
pip install .
```


📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

🤝 Contributing
Pull requests are welcome!
If you have suggestions for optimizers, metrics, or additional features, feel free to open an issue.


## 📊 KnnOpt accuracy comparison with other models

### Case name: diabetes regression
This case uses sklearn diabetes dataset. All models use test_size: 0.25, random_state: 42

-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE% | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     |  target |23.67% | 25.52% |n_neighbours=5, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    |  target |26.36% | 26.03% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        |  target |24.50% | 27.50% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest| target |25.07% | 24.38% |n_estimators=300,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         | target |23.96% | 27.46% |hidden_layer_sizes: [50, 30, 20], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  target |26.25% | 27.15% |n_neighbours=5, weights: distance, metric: euclidean|



### Case name: california_housing regression
It uses sklearn california_housing dataset. All models use test_size: 0.25, random_state: 42

-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     |  target |13.82% | 13.57% |n_neighbours=5, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    |  target |15.42% | 15.05% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        |  target |23.69% | 24.37% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest| target |22.38% | 22.24% |n_estimators=300,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         | target |20.64% | 20.69% |hidden_layer_sizes: [50, 30, 20], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  target |21.93% | 22.40% |n_neighbours=5, weights: distance, metric: euclidean|


### Case name: heat_tr_coef regression

The dataset comes from CFD simulation of heat transfer coefficients of two-dimensional rectangular objects
of different dimensions and flow configs. The purpose here is to use the dataset to validate different models.
All models use test_size: 0.3, random_state: 42


-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     |  hcoef |7.18% | 8.22% |n_neighbours=5, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    |  hcoef |9.24% | 8.11% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        |  hcoef  |5.61% | 7.28% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest| hcoef |10.81% | 8.21% |n_estimators=300,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         | hcoef  |33.38% | 32.46% |hidden_layer_sizes: [50, 30, 20], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  hcoef |14.76% | 10.81% |n_neighbours=5, weights: distance, metric: euclidean|



### Case name: metal_strength regression

The dataset correlates metal strength properties with different operational treatment conditions.
There are 4 targets in this case. The physics background of the dataset is obsecured, but the data relationships are kept for model
validations. All models use test_size: 0.2, random_state: 42


-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     |t1,t2,t3,t4 |6.57.2.02,5.10,8.63% |6.51,2.28,11.31,12.08% |n_neighbours=3, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | t1,t2,t3,t4 |7.84,2.09,6.2,8.19% | 7.99,1.87,11.11,12.42% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        |  t1,t2,t3,t4  |18.96,2.65,11.23,15.02% |20.96,3.29,15.23,16.91% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest| t1,t2,t3,t4 |6.84,1.99,5.70,8.18% |6.82,2.09,8.62,11.68% |n_estimators=300,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         | t1,t2,t3,t4 |17.74,14.24,26.39,28.59% |18.81,7.43,31.58,18.72% |hidden_layer_sizes: [50, 30, 20], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  t1,t2,t3,t4  |10.14,2.13,8.71,11.10% | 8.51,2.80,12.05,10.93% |n_neighbours=5, weights: distance, metric: euclidean|

### Case name: synthetic regression

The dataset is generated using sklearn make_regression function with some post transformation. All models use test_size: 0.3,
random_state: 42, total 1000 records are generated, of which 10% is used for validation and 90% for train/test, and 30% of the remaining data is for test.
The target is of normal distribution with mean=10, std=1, noise level 10%.


-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     |target |2.58% |2.75% |n_neighbours=3, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | target |2.13% | 2.11% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        |  target |0.05% |0.06% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest| target |3.53% |3.46% |n_estimators=300,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         |target |0.31% |0.29% |hidden_layer_sizes: [50, 30, 20], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  target |3.76 | 3.92% |n_neighbours=3, weights: distance, metric: euclidean|


### Case name: product quality reqression

This case correlates the product quality parameters with its raw material composition. All  models use test_size=0.3.
4 product characteristics are chosen for comparing the model prediction capabilities: size, strength, component6, react_rate1


-----------------------------------------
| Model name | Target Name| Test Rel MAE% |Validate Rel MAE | Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     | size, strength, component6, react_rate1|1.87,0.57,2.81,2.47% |3.49,0.67,1.88,3.18% |n_neighbours=3, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | size, strength, component6, react_rate1|5.07,3.51,3.57,3.15% | 5.93,2.43,2.59,3.31% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVR        | size, strength, component6, react_rate1 |3.65,0.75,3.68,3.15% |3.12,0.79,2.86,3.15% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest|size, strength, component6, react_rate1 |4.96,3.48,3.36,2.65% |7.04,4.31,2.55,3.28% |n_estimators=200,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP         |size, strength, component6, react_rate1|8.12,6.33,11.87,6.52% |5.22,8.61,11.62,7.57% |hidden_layer_sizes: [100], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn       |  size, strength, component6, react_rate1|2.29,0.68,3.57,3.39% |3.59,0.69,2.05,3.29% |n_neighbours=3, weights: distance, metric: euclidean|


### Case name: breast_cancer classfication

This case uses sklearn breast_cancer dataset.  All  models use test_size=0.3, random_state: 42

-----------------------------------------
| Model name | Target Name| Test Accuracy% |Validate Accuracy| Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     | target|98.72% |100% |n_neighbours=5, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | target|97.44% | 100% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVC       | target |98.72% |98.25% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest|target |95.12% |96.49% |n_estimators=200,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP        |target|98.37% |98.25% |hidden_layer_sizes: [100], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn        | target|96.75% |94.74% |n_neighbours=3, weights: distance, metric: euclidean|



### Case name: iris classification

This case uses sklearn iris dataset.  All  models use test_size=0.3, random_state: 42

-----------------------------------------
| Model name | Target Name| Test Accuracy% |Validate Accuracy| Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     | target|100% |100% |n_neighbours=5, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | target|97.50% | 100% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVC       | target |96.97% |100% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest|target |84.85% |100% |n_estimators=200,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP        |target|96.97% |93.33% |hidden_layer_sizes: [100], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn        | target|87.88% |100% |n_neighbours=3, weights: distance, metric: euclidean|




### Case name: wine classification

This case uses sklearn wine dataset.  All  models use test_size=0.3, random_state: 42


-----------------------------------------
| Model name | Target Name| Test Accuracy% |Validate Accuracy| Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     | target|100% |100% |n_neighbours=4, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | target|97.92% | 100% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVC       | target |87.18% |100% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest|target |97.44% |100% |n_estimators=200,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP        |target|92.31% |100% |hidden_layer_sizes: [100], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn        | target|89.74% |94.44% |n_neighbours=4, weights: distance, metric: euclidean|


### Case name: synthetic_classification 

 It uses sklearn make_classification to generate the dtaset.  1000 records are generated,
of which 900 is for train/test the model, and 100 for validate the model. There are 10 features and 1 target in this case. 
All  models use test_size=0.2, random_state: 42


-----------------------------------------
| Model name | Target Name| Test Accuracy% |Validate Accuracy| Model parameters|
|------------|---------|----------|----------------|----------------|
| KnnOpt     | target|92.22% |85% |n_neighbours=7, weights: distance, metric: euclidean, optimizer: LN_COBYLA|
| XGBoost    | target|93.33% | 84% |n_estimators=300,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8|
| SVC       | target |88.89% |81% |C=200, gamma=0.05,epsilon=0.01,Kernel=RBF|
| RandomForest|target |92.78% |85% |n_estimators=200,max_depth=6,min_samples_split=2,min_samples_leaf=1|
| MLP        |target|88.89% |81% |hidden_layer_sizes: [100], activation: relu, solver: adam, alpha: 0.0001, learning_rate: constant, max_iter: 600|
| Knn        | target|83.33% |80% |n_neighbours=4, weights: distance, metric: euclidean|


## Concluding remarks

- KnnOpt model consistently shows  stable and mostly better accuracy compare to other models both reegression and classification cases.
- The consistently good performance is because KnnOpt introduces a weights-based importance
factor to measure a feature with regard to a target. The weights matrix is then optimized to minimize the 
RMAE error for unseen dataset in the training process. If the unseen dataset covers a wide range of 
potential new data in prediction, it is then able to give good accuracy
- The accuracy of the model depends on how close of a new data record is from the train/test data. If a data 
record is too far away from the dataset used to train/test the model, the prediction accuracy will drop.

