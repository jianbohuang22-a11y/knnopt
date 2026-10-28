train-validate-split combined_csv.csv 
knn-train train_data.csv -f f1,f2,f3,f4,f5 -t target1,target2,target3,target4 -n 5 --test-size 0.2
knnopt-validate reg ./ validation_input.csv
