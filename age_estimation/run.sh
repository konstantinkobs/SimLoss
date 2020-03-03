#!/bin/bash

run_experiment () {
    DATASET=$1
    mkdir -p "results_$DATASET"
    for REDUCTION_FACTOR in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for NUMBER in {0..9}
        do
            python3 run.py --reductionfactor $REDUCTION_FACTOR --number $NUMBER --dataset $DATASET > "results_$DATASET/${REDUCTION_FACTOR}_${NUMBER}.out" 
        done
    done
}

run_experiment AFAD
run_experiment UTKFace
