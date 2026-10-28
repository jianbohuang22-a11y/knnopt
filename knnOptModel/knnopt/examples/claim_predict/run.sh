train-validate-split combined_csv.csv 
knn-train train_data.csv -f  f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17 \
-t target --model-type cls -n 5 --test-size 0.3
knnopt-validate cls ./ validation_input.csv
