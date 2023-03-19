## Docker setup 
docker run -it --gpus all -v $(pwd):/usr/home/workspace continuumio/miniconda3

conda install python=3.8.10

#pip install tensorflow-2.7.4-cp38-cp38-manylinux2010_x86_64.whl
#pip install tensorflow-gpu==2.7.4
pip install -r requirements_bark.txt

apt-get update
apt-get install -y xvfb

#apt-get install python3-opencv

## Train
cd usr/home/workspace/
python ris_image_train_bark_ml.py --exp_name bark_ml_ex_34 --replay_buffer_size 50000 --start_timesteps 20000 --eval_freq 2000 --batch_size 2048 --max_episode_length 300 --state_dim 5 --max_timesteps 300000

## Validate
cd usr/home/workspace/
### make VALIDATE_ENV = True in custom_bark_gym_env/custom_gym_bark_ml_env.py (for videos)
python ris_image_validate_bark_ml.py --exp_name bark_ml_ex_14 --start_timesteps 0 --state_dim 5 --max_timesteps 100 --no_video True

python utilite_video_generator.py 