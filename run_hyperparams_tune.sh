#!/bin/bash

# HyperParams
folder_name="hyperparam_tune"
run=0
start_run=1
max_timesteps=200000
state_dims=(20 80) # 2
Lambdas=(10 1 0.1 0.01 0.001) # 5
alphas=(1 0.1 0.01) # 3
n_ensembles=(5 10 25) # 3

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
                exp_name="$folder_name""_"$run
                echo "exp_name: $exp_name"
                if [ $run -ge $start_run ]; then
                    python ris_train_polamp_env.py --exp_name $exp_name --max_timesteps $max_timesteps --state_dim $state_dim --Lambda $Lambda --alpha $alpha --n_ensemble $n_ensemble
                fi
            done
        done
    done
done

