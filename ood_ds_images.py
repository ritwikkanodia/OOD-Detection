from models import *

from PIL import Image
import torch
import os
import numpy as np
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
import cv2
from torch.utils.data import DataLoader
import gc
from helper import maskAndResize, one_image_mean, batch_image_mean


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
ood_ds_feats = np.array([])

for img_name in tqdm.tqdm(kitti_val_images):
    label_dir = root + "/kitti_val/kitti_labels/" + img_name[:-1].split(".")[0] + ".txt"
    image_dir = root + "/kitti_val/images/" + img_name
    masked_image_dir = root + "/kitti_val/ood_ds_masked_images/"
    image = cv2.imread(image_dir)
    if count == 0:
        print("For image:")
        print(label_dir)
        print(image_dir)
        print(masked_image_dir)
        print()
    image_batch = np.array([])
    bbox_count = 0
    with open(label_dir) as fp:
        Lines = fp.readlines()

        for line in Lines:
            label = line.split(" ")[0]
            if label == "Car" or label == "Van":
                continue
            print("for bbox ", count, " containing ", line.split(" ")[0])
            bbox_coords = line.split(" ")[4:8]
            x1, y1, x2, y2 = (
                math.floor(float(bbox_coords[0])),
                math.floor(float(bbox_coords[1])),
                math.floor(float(bbox_coords[2])),
                math.floor(float(bbox_coords[3])),
            )

            # masking
            masked_img = maskAndResize(image, x1, y1, x2, y2)
            cv2.imwrite(
                masked_image_dir
                + img_name
                + "_"
                + label
                + "_"
                + str(bbox_count)
                + ".png",
                masked_img,
            )
            count += 1
            bbox_count += 1

            # aggregating all bbox in an img as a batch
            if image_batch.shape[0] > 0:
                image_batch = np.append(
                    image_batch, masked_img.reshape((1, 3, 416, 416)), 0
                )
            else:
                image_batch = masked_img.reshape((1, 3, 416, 416))

            print("image_batch.shape: ", image_batch.shape)
            del masked_img

    if image_batch.shape[0] > 0:
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
        if ood_ds_feats.shape[0] > 0:
            ood_ds_feats = np.append(ood_ds_feats, inter_features_mean_batch, 0)
        else:
            ood_ds_feats = inter_features_mean_batch

        print("ood_ds_feats.shape: ", ood_ds_feats.shape)
        del (
            image_batch,
            prediction_batch,
            inter_features_mean_batch,
            inter_features_batch,
        )
        gc.collect()
        torch.cuda.empty_cache()
    # if count > 12000:
    #     print("ENOUGH IMAGES ALR")
    #     break


print("Saving feats")
np.save(
    "/home/FYP/ritwik002/main/oodds/oodds_features/module_0/conv_0/oodds.npy",
    ood_ds_feats,
)

