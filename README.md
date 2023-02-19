## Docker setup 
docker run -it --gpus all -v $(pwd):/usr/home/workspace continuumio/miniconda3

conda install python=3.8.10

cd usr/home/workspace/
#pip install tensorflow-2.7.4-cp38-cp38-manylinux2010_x86_64.whl
#pip install tensorflow-gpu==2.7.4
pip install -r requirements_bark.txt

apt-get update
apt-get install -y xvfb

## Train
python ris_image_train_bark_ml.py --exp_name bark_ml_ex_12 --replay_buffer_size 50000 --start_timesteps 20000 --eval_freq 2000 --batch_size 256 --max_episode_length 250 --state_dim 64

## Validate
python ris_image_validate_bark_ml.py --exp_name bark_ml_ex_1 --start_timesteps 0 --state_dim 64

python utilite_video_generator.py 