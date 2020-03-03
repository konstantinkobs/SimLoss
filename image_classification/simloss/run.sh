#!/bin/bash

run_experiment () {
    mkdir -p "results"
    for LOWER_BOUND in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
    do
        for NUMBER in {0..9}
        do
            python3 run.py --lower-bound $LOWER_BOUND --run $NUMBER > "results/${LOWER_BOUND}_${NUMBER}.out" 
        done
    done
}

run_experiment
