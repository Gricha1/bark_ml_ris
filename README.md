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



# polamp env

# train PPO
python origin_train_lagrangian_ppo.py --name_save test_PPO_2

# goal polamp env
cd usr/home/workspace/
## Train RIS_PPO
python train_ris_ppo_polamp.py --name_save test_RIS_PPO_polamp_ex51

## Validate RIS_PPO ( not in docker )
### change --training to 0
python train_ris_ppo_polamp.py --name_val test_RIS_PPO_polamp_ex51

python utilite_video_generator.py 