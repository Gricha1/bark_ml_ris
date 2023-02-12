apt-get install -y xvfb


docker run -it --gpus all -v $(pwd):/usr/home/workspace continuumio/miniconda3