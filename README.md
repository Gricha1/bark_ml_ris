# Docker setup 
docker run -it --gpus all -v $(pwd):/usr/home/workspace continuumio/miniconda3
cd /usr/home/workspace

conda install python=3.8.10

#pip install tensorflow-2.7.4-cp38-cp38-manylinux2010_x86_64.whl
#pip install tensorflow-gpu==2.7.4
pip install -r requirements_bark.txt

apt-get update
apt-get install -y xvfb

#apt-get install python3-opencv

# Ant env(origin)
cd usr/home/workspace/
## Train Ant U
python origin_train_ant.py --exp_name test_origin_ant_run_1

## Validate Ant U
python origin_validate_ant.py --exp_name test_origin_ant_run_1


# bark_ml env
cd usr/home/workspace/
## Train bark_ml
python ris_image_train_bark_ml.py --exp_name bark_ml_ex_121 --replay_buffer_size 100000 --start_timesteps 20000 --eval_freq 2000 --batch_size 512 --max_episode_length 300 --state_dim 5 --wandb_project RIS_bark_ml_train

## Validate bark_ml
#### make VALIDATE_ENV = True in custom_bark_gym_env/custom_gym_bark_ml_env.py (for videos)
python ris_image_validate_bark_ml.py --exp_name bark_ml_ex_14 --start_timesteps 0 --state_dim 5 --max_episode_length 100 --no_video True


# polamp env
cd usr/home/workspace/
## Train polamp_env
python ris_train_polamp_env.py --exp_name polamp_env_ex_1 --wandb_project RIS_polamp_env_train

## Validate polamp_env ( not in docker )
python ris_validate_polamp_env.py --exp_name polamp_env_ex_1 --start_timesteps 0 --no_video True

python utilite_video_generator.py 