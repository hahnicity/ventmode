#!/bin/bash
echo "Run Hyperparameter Search"
echo "run hyperparameters with logistic reg"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo log_reg --use-train-only --grid-search
echo "run hyperparameters with random forest"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo rf --use-train-only --grid-search
echo "run hyperparameters with MLP"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo mlp --use-train-only --grid-search
echo "run hyperparameters with SVM"
python main.py -p dataset-repro.pkl --split-type cross_pt --algo svm --use-train-only --grid-search
