
# origin envs

## train
python origin_main.py --env_name AntMazeSparse --model_dir ex_1 

## validate
python origin_eval.py --env_name AntMazeSparse --model_dir ./pretrained_models

## logger
tensorboard --logdir logs/hrac --bind_all


# polamp env

## train
python polamp_main.py --env_name AntMazeSparse --model_dir polamp_ex_1 

## validate
python polamp_eval.py --env_name AntMazeSparse --model_dir polamp_ex_1

## logger
tensorboard --logdir logs/hrac --bind_all
