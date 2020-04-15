#!/bin/bash

# First reproduce our main results. This may take awhile while the code calculates
# feature values for the model
echo "Run main experiment (Table 3) with Random Forest"
python main.py --to-pickle dataset-repro.pkl --split-type validation --with-lookahead-conf 50 --lookahead-conf-frac .6

# Run experiments to evaluate whether to use lookahead/lookbehind and which parameterization to use
#
echo "Run Lookbehind Experiments"
echo "Run look-behind window length 10, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-majority-window 10
echo "Run look-behind window length 20, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-majority-window 20
echo "Run look-behind window length 30, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-majority-window 30
echo "Run look-behind window length 40, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-majority-window 40
echo "Run look-behind window length 50, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-majority-window 50
echo "Run look-behind window length 10, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 10  --conf-window-frac .6
echo "Run look-behind window length 10, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 10 --conf-window-frac .7
echo "Run look-behind window length 10, fraction: .8"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 10 --conf-window-frac .8
echo "Run look-behind window length 10, fraction: .9"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 10 --conf-window-frac .9
echo "Run look-behind window length 20, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 20  --conf-window-frac .6
echo "Run look-behind window length 20, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 20 --conf-window-frac .7
echo "Run look-behind window length 20, fraction: .8"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 20 --conf-window-frac .8
echo "Run look-behind window length 20, fraction: .9"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 20 --conf-window-frac .9
echo "Run look-behind window length 30, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 30  --conf-window-frac .6
echo "Run look-behind window length 30, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 30 --conf-window-frac .7
echo "Run look-behind window length 30, fraction: .8"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 30 --conf-window-frac .8
echo "Run look-behind window length 30, fraction: .9"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 30 --conf-window-frac .9
echo "Run look-behind window length 40, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 40  --conf-window-frac .6
echo "Run look-behind window length 40, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 40 --conf-window-frac .7
echo "Run look-behind window length 40, fraction: .8"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 40 --conf-window-frac .8
echo "Run look-behind window length 40, fraction: .9"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 40 --conf-window-frac .9
echo "Run look-behind window length 50, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 50  --conf-window-frac .6
echo "Run look-behind window length 50, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 50 --conf-window-frac .7
echo "Run look-behind window length 50, fraction: .8"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 50 --conf-window-frac .8
echo "Run look-behind window length 50, fraction: .9"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-conf-window 50 --conf-window-frac .9

echo "Run lookahead experiments"
echo "Run look-ahead window length 10, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 10  --lookahead-conf-frac .5
echo "Run look-ahead window length 10, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 10  --lookahead-conf-frac .6
echo "Run look-ahead window length 10, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 10 --lookahead-conf-frac .7
echo "Run look-ahead window length 20, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 20  --lookahead-conf-frac .5
echo "Run look-ahead window length 20, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 20  --lookahead-conf-frac .6
echo "Run look-ahead window length 20, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 20 --lookahead-conf-frac .7
echo "Run look-ahead window length 30, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 30  --lookahead-conf-frac .5
echo "Run look-ahead window length 30, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 30  --lookahead-conf-frac .6
echo "Run look-ahead window length 30, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 30 --lookahead-conf-frac .7
echo "Run look-ahead window length 40, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 40  --lookahead-conf-frac .5
echo "Run look-ahead window length 40, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 40  --lookahead-conf-frac .6
echo "Run look-ahead window length 40, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 40 --lookahead-conf-frac .7
echo "Run look-ahead window length 50, fraction: .5"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 50  --lookahead-conf-frac .5
echo "Run look-ahead window length 50, fraction: .6"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 50  --lookahead-conf-frac .6
echo "Run look-ahead window length 50, fraction: .7"
python main.py -p dataset-repro.pkl --split-type cross_pt --use-train-only --with-lookahead-conf 50 --lookahead-conf-frac .7

# Run hyperparameter search. This can take awhile, especially for SVM
#
echo "Run Hyperparameter Search"
echo "run hyperparameters with logistic reg"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo log_reg --use-train-only --grid-search
echo "run hyperparameters with random forest"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo rf --use-train-only --grid-search
echo "run hyperparameters with MLP"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo mlp --use-train-only --grid-search
echo "run hyperparameters with SVM"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo svm --use-train-only --grid-search

# Next you can run to reproduce results on other algorithms such as MLP/SVM/LogReg
#
# These results should be faster by virtue of the fact you've already saved your
# dataset to a pickle file
echo "Run Algorithm Evaluation Experiments"
echo "Run with Neural Network"
python main.py -p dataset-repro.pkl --split-type cross_pt --with-lookahead-conf 50 --lookahead-conf-frac .6 --algo mlp --use-train-only
echo "Run with SVM"
python main.py -p dataset-repro.pkl --split-type cross_pt --with-lookahead-conf 50 --lookahead-conf-frac .6 --algo svm --use-train-only
echo "Run with Logistic Regression"
python main.py -p dataset-repro.pkl --split-type cross_pt --with-lookahead-conf 50 --lookahead-conf-frac .6 --algo log_reg --use-train-only
echo "Run with Random Forest"
python main.py -p dataset-repro.pkl --split-type cross_pt --with-lookahead-conf 50 --lookahead-conf-frac .6 --algo rf --use-train-only
echo "Run with LSTM"
python torch_lstm.py dataset-repro.pkl --shuffle

# Create all dataframes for use in random ablation before we process results
./random_data_removal.sh

# Run random ablation experiments for various algos
python dataset_ablation_collate.py dataset-repro.pkl random --algo rf -t 4
python dataset_ablation_collate.py dataset-repro.pkl random --algo mlp -t 4
python dataset_ablation_collate.py dataset-repro.pkl random --algo log_reg -t 4
python dataset_ablation_collate.py dataset-repro.pkl random --algo svm -t 4

# Run Optimal Ablation By Only utilizing first N observations for ventmode in a file.
python dataset_size_reduction.py -p dataset-repro.pkl -t sensitivity
python dataset_size_reduction.py -p dataset-repro.pkl -t optimal

# Run visualizations for any results
python visualize_ablation_results.py random_ablation_results_rf.pkl -x ablation -a "Random Forest" -t random
python visualize_ablation_results.py random_ablation_results_mlp.pkl -x ablation -a "MLP" -t random
python visualize_ablation_results.py random_ablation_results_svm.pkl -x ablation -a "SVM" -t random
python visualize_ablation_results.py random_ablation_results_log_reg.pkl -x ablation -a "Logistic Regression" -t random

# Visualize size reduction results
python visualize_ablation_results.py contiguous-ablation-results.pkl -x n -t size_reduction -a "Random Forest"
