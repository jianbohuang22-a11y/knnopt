knnopt-train ./train_data.csv -f f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11 \
-t medical_event --model-type cls  -n  7 --test-size 0.25
pt-validate cls ./ validate_input.csv
