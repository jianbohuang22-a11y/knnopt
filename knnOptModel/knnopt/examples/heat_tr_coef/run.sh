train-validate-split data.csv
knn-train train_data.csv -f a,b,c,a1,d,f,f0,e,Tobj,fv,Ti -t hcoef,Tmax,Tmin,Tmean,Tmedian,Tstd -n 5 --test-size 0.3
knnopt-validate reg ./ validation_input.csv

