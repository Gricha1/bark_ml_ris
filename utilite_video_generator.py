import cv2
import os
import argparse
from pathlib import Path
import pathlib
import wandb
import numpy as np

def create_video(image_folder, result_video_name, run):
  try:
    sorted_img_num_pairs = sorted([(int(img[3:][:-4]), img) for img in os.listdir(image_folder) if img.endswith(".png")])
  except:
    sorted_img_num_pairs = sorted([(int(img[:-4]), img) for img in os.listdir(image_folder) if img.endswith(".png")])  
  images = [pair[1] for pair in sorted_img_num_pairs]

  video = None
  for image in images:
    np_img = np.expand_dims(cv2.imread(os.path.join(image_folder, image)), 0)
    if video is None:
      video = np_img
    else:
      video = np.concatenate([video, np_img], axis=0)

  video = np.moveaxis(video, [0, 1, 2, 3], [0, 2, 3, 1])
  run.log({f"{result_video_name}": wandb.Video(video, fps=20)})
  print("video is created:", video.shape)


def generate_video(env=False, obs=False, run=None):
  if run is None:
    run = wandb.init(project='RIS_bark_ml_validate')

  working_dir_name = str(pathlib.Path().resolve())
  if env:
    image_folder = working_dir_name+"/video_validation/pngs"
    result_video_name = "env_video"
    create_video(image_folder, result_video_name, run)

  if obs:
    image_folder = working_dir_name+"/video_validation/obs_pngs"
    result_video_name = "obs_video"
    create_video(image_folder, result_video_name, run)

if __name__ == "__main__":
  generate_video(env=True, obs=False)
  