#!/bin/bash

# HyperParams
folder_name="hyperparam_tune"
run=0
max_timesteps=200000
state_dims=(20 80)
Lambdas=(10 1 0.1 0.01 0.001)
alphas=(1 0.1 0.01)
n_ensembles=(5 10 25)

# cycle for HyperParams tune
for state_dim in "${state_dims[@]}"
do
    for Lambda in "${Lambdas[@]}"
    do
        for alpha in "${alphas[@]}"
        do
            for n_ensemble in "${n_ensembles[@]}"
            do
                run=$((run + 1))
                echo "run: $run"
                python ris_train_polamp_env.py --exp_name $folder_name_$run --max_timesteps $max_timesteps --state_dim $state_dim --Lambda $Lambda --alpha $alpha --n_ensemble $n_ensemble
            done
        done
    done
done

