import cv2
import os
import argparse
from pathlib import Path
import pathlib

def create_video(image_folder, result_video_name):
  try:
    sorted_img_num_pairs = sorted([(int(img[3:][:-4]), img) for img in os.listdir(image_folder) if img.endswith(".png")])
  except:
    sorted_img_num_pairs = sorted([(int(img[:-4]), img) for img in os.listdir(image_folder) if img.endswith(".png")])
  #images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
  images = [pair[1] for pair in sorted_img_num_pairs]
  frame = cv2.imread(os.path.join(image_folder, images[0]))
  height, width, layers = frame.shape

  video = cv2.VideoWriter(result_video_name, 0, 15, (width,height))

  for image in images:
      video.write(cv2.imread(os.path.join(image_folder, image)))

  cv2.destroyAllWindows()
  video.release()

def generate_video(env=False, obs=False, name="0"):
  working_dir_name = str(pathlib.Path().resolve())
  #image_folder = '/content/pngs/run_1'
  if env:
    image_folder = working_dir_name+"/video_validation/pngs/run_1"
    result_video_name = working_dir_name+"/video_validation/"+ f'video{name}.avi'
    create_video(image_folder, result_video_name)

  if obs:
    #image_folder = '/content/obs_pngs'
    image_folder = working_dir_name+"/video_validation/obs_pngs"
    result_video_name = working_dir_name+"/video_validation/" + f'obs_video{name}.avi'
    create_video(image_folder, result_video_name)