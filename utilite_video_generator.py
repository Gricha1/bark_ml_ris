import cv2
import os
import argparse
from pathlib import Path
import pathlib
#from google.colab import drive


parser = argparse.ArgumentParser()
working_dir_name = str(pathlib.Path().resolve())
parser.add_argument("--obs", default=working_dir_name+"/video_validation/obs_pngs")
parser.add_argument("--env", default=working_dir_name+"/video_validation/pngs/run_1")
args = parser.parse_args()

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


#image_folder = '/content/pngs/run_1'
image_folder = args.env
result_video_name = working_dir_name+"/video_validation/"+ 'video.avi'
create_video(image_folder, result_video_name)

#image_folder = '/content/obs_pngs'
image_folder = args.obs
result_video_name = working_dir_name+"/video_validation/" + 'obs_video.avi'
create_video(image_folder, result_video_name)