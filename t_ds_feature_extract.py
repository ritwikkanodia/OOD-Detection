from models import *

from PIL import Image
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
import datetime
import cv2
from torch.utils.data import DataLoader
import gc


# helper functions
def maskAndResize(image, x1, y1, x2, y2):
    newsize = (416, 416)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    resized = cv2.resize(masked, newsize)
    return resized


def one_image_mean(single_img):
    # w, h, d = single_img.shape[0], single_img.shape[1], single_img.shape[2]
    # single_img = single_img.reshape((d, w, h))
    return [channel.mean() for channel in single_img]


def batch_image_mean(batch_images):
    batch_img_mean = []
    for single_img in batch_images:
        img_mean = one_image_mean(single_img)
        batch_img_mean.append(img_mean)
    return np.array(batch_img_mean)


# load model
model_def = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
# model_def = "yolov3-custom.cfg"
img_size = 416
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_weights = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
# pretrained_weights = "yolov3_ckpt_48.pth"

model = Darknet(model_def, img_size=img_size).to(device)
model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cpu")))
print("Model Loaded")

# register hook
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


model.module_list[0].conv_0.register_forward_hook(get_features("feats"))

print("Hook registered for layer conv_0")

root = "/home/FYP/ritwik002/data"
# root = "data"
kitti_val_images = os.listdir(root + "/kitti_val/images")[1:]
count = 0
id_ds_feats = np.array([])

for img_name in tqdm.tqdm(kitti_val_images):
    label_dir = (
        root + "/kitti_val_detection/labels/" + img_name[:-1].split(".")[0] + ".txt"
    )
    image_dir = root + "/kitti_val/images/" + img_name
    masked_image_dir = root + "/kitti_val/masked_images_resized/"
    image = cv2.imread(image_dir)
    print("For image:")
    print(label_dir)
    print(image_dir)
    print(masked_image_dir)
    print()
    image_batch = np.array([])
    with open(label_dir) as fp:
        Lines = fp.readlines()
        if len(Lines) != 0:
            for line in Lines:
                print("for bbox ", count)
                bbox_coords = line.split(",")
                print(bbox_coords)
                bbox_coords = [math.floor(float(x)) for x in bbox_coords]
                x1, x2, y1, y2 = (
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                )

                # masking
                masked_img = maskAndResize(image, x1, y1, x2, y2)
                cv2.imwrite(masked_image_dir + str(count) + ".png", masked_img)
                count += 1

                # aggregating all bbox in an img as a batch
                if image_batch.shape[0] > 0:
                    image_batch = np.append(
                        image_batch, masked_img.reshape((1, 3, 416, 416)), 0
                    )
                else:
                    image_batch = masked_img.reshape((1, 3, 416, 416))

                print("image_batch.shape: ", image_batch.shape)
                del masked_img

            # prediction
            image_batch = torch.tensor(image_batch).float()
            prediction_batch = model(image_batch.cuda())
            inter_features_batch = features["feats"]  # (batch, 32, 416, 416)
            inter_features_batch = np.array(inter_features_batch.cpu())
            inter_features_mean_batch = batch_image_mean(
                inter_features_batch
            )  # (batch, 32)
            # print("inter_features_batch: ", inter_features_batch.shape)
            # print("inter_features_mean_batch: ", inter_features_mean_batch.shape)
            if id_ds_feats.shape[0] > 0:
                id_ds_feats = np.append(id_ds_feats, inter_features_mean_batch, 0)
            else:
                id_ds_feats = inter_features_mean_batch

            print("id_ds_feats.shape: ", id_ds_feats.shape)
            del (
                image_batch,
                prediction_batch,
                inter_features_mean_batch,
                inter_features_batch,
            )
            gc.collect()
            torch.cuda.empty_cache()


print("Saving feats")
np.save(
    "/home/FYP/ritwik002/main/idds/idds_features/module_0/conv_0/idds.npy", id_ds_feats
)
