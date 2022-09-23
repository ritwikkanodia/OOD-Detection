from models import *

import torch
import numpy as np
import gc
from tqdm import tqdm
import numpy as np
from numba import cuda


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


model_def = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
# model_def = "yolov3-custom.cfg"
img_size = 416
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_weights = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
# pretrained_weights = "yolov3_ckpt_48.pth"

model = Darknet(model_def, img_size=img_size).to(device)
model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cpu")))
print("Model Loaded")

features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


model.module_list[0].conv_0.register_forward_hook(get_features("feats"))

print("Hook registered for layer conv_0")


tds_images = np.load("/home/FYP/ritwik002/main/tds/tds_images/tds_images_4.npy")
# tds_images_1 = np.load("tds_images/tds_images_1.npy")
# all_images = torch.tensor(all_images).to("cpu")
print("Loaded all images np array with shape: ", tds_images.shape)


print("getting features")
td_ds_feats = np.array([])
batch_size = 16
for batch in tqdm(range(0, tds_images.shape[0], batch_size)):
    image_batch = tds_images[batch : batch + batch_size]  # (batch, 3, 416, 416)
    image_batch = torch.tensor(image_batch).float()
    prediction_batch = model(image_batch.cuda())

    inter_features_batch = features["feats"]  # (batch, 32, 416, 416)
    inter_features_batch = np.array(inter_features_batch.cpu())
    inter_features_mean_batch = batch_image_mean(inter_features_batch)  # (batch, 32)
    # print("inter_features_batch: ", inter_features_batch.shape)
    # print("inter_features_mean_batch: ", inter_features_mean_batch.shape)
    if td_ds_feats.shape[0] > 0:
        td_ds_feats = np.append(td_ds_feats, inter_features_mean_batch, 0)
    else:
        td_ds_feats = inter_features_mean_batch

    print(td_ds_feats.shape)
    del image_batch, prediction_batch, inter_features_mean_batch, inter_features_batch
    gc.collect()
    torch.cuda.empty_cache()

print("Saving feats")
np.save(
    "/home/FYP/ritwik002/main/tds/tds_features/module_0/conv_0/tds_4.npy", td_ds_feats
)
# np.save("tds_features/tds_feats_0_conv_0_4.npy", td_ds_feats)

print("DONE!")
