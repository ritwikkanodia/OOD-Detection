import os
import cv2
import numpy as np
from tqdm import tqdm

images_dir = os.listdir('data/masked_images_resized')
all_images = np.array([])
for image_dir in tqdm(images_dir):
    img_arr = cv2.imread('data/masked_images_resized/' + image_dir)
    img_arr = img_arr.reshape((1, 416, 416, 3))
    if all_images.shape[0]>0:
        all_images = np.append(all_images, img_arr, 0)
    else:
        all_images = img_arr
    del img_arr
    
        
print(all_images.shape)
np.save('all_images.npy', all_images)
