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


# polamp env
cd usr/home/workspace/
## Train polamp_env
python ris_train_polamp_env.py --exp_name polamp_env_ex_1 --wandb_project RIS_polamp_env_train

## Validate polamp_env ( not in docker )
python ris_validate_polamp_env.py --exp_name polamp_env_ex_1 --no_video True --wandb_project RIS_polamp_env_validate

python utilite_video_generator.py 