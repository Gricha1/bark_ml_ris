import cv2
import os

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


image_folder = '/content/pngs/run_1'
result_video_name = 'video.avi'
create_video(image_folder, result_video_name)

image_folder = '/content/obs_pngs'
result_video_name = 'obs_video.avi'
create_video(image_folder, result_video_name)