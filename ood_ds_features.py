from models import *
import os
import torch
import numpy as np
import gc
from tqdm import tqdm
import numpy as np
from numba import cuda

from helper import one_image_mean, batch_image_mean, load_model


"""
model_def = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
# model_def = "yolov3-custom.cfg"
img_size = 416
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_weights = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
# pretrained_weights = "yolov3_ckpt_48.pth"

model = Darknet(model_def, img_size=img_size).to(device)
model.load_state_dict(torch.load(pretrained_weights, map_location=torch.device("cpu")))
print("Model Loaded")
"""


def extract_features_layer(model, data_type, module, batch_size):
    layer = "conv_2"
    print("Extracting features for module ", module, " and layer ", layer)
    # register the hook
    features = {}

    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()

        return hook

    model.module_list[module].conv_2.register_forward_hook(get_features("feats"))

    # load images
    images_dir = (
        "/home/FYP/ritwik002/main/"
        + data_type
        + "/"
        + data_type
        + "_images/"
        + data_type
        + "_images.npy"
    )
    ds_images = np.load(images_dir)
    print("loaded images from: ", images_dir)

    # register paths
    module_dir = (
        "/home/FYP/ritwik002/main/"
        + data_type
        + "/"
        + data_type
        + "_features/module_"
        + str(module)
    )
    feature_dir = module_dir + "/" + layer
    feature_filename = data_type + "_features.npy"
    print("Storing features at: ", os.path.join(feature_dir, feature_filename))
    if not os.path.isdir(module_dir):
        os.mkdir(module_dir)

    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)

    print("feature_dir: ", feature_dir)
    print("feature_filename: ", feature_filename)

    # extract features
    print("Extracting feats")
    ds_feats = np.array([])
    for batch in tqdm(range(0, ds_images.shape[0], batch_size)):
        image_batch = ds_images[batch : batch + batch_size]  # (batch, 3, 416, 416)
        image_batch = torch.tensor(image_batch).float()
        prediction_batch = model(image_batch.cuda())

        inter_features_batch = features["feats"]  # (batch, 32, 416, 416)
        inter_features_batch = np.array(inter_features_batch.cpu())
        inter_features_mean_batch = batch_image_mean(
            inter_features_batch
        )  # (batch, 32)
        if ds_feats.shape[0] > 0:
            ds_feats = np.append(ds_feats, inter_features_mean_batch, 0)
        else:
            ds_feats = inter_features_mean_batch

        print(ds_feats.shape)
        del (
            image_batch,
            prediction_batch,
            inter_features_mean_batch,
            inter_features_batch,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # saving feats

    print("Saving feats")
    np.save(os.path.join(feature_dir, feature_filename), ds_feats)


# load model
config_dir = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
pretrained_weights_dir = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
model = load_model(config_dir, pretrained_weights_dir)

# register hook
module = 2
layer = "conv_2"

print("Extracting features for module ", module, " and layer ", layer)
features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


model.module_list[module].conv_2.register_forward_hook(get_features("feats"))

print("Hook registered for layer conv_2")

#
id_ds_images = np.load("/home/FYP/ritwik002/main/idds/idds_images/idds_images.npy")
print("Loaded idds images np array with shape: ", id_ds_images.shape)


module_dir = "/home/FYP/ritwik002/main/idds/idds_features/module_" + str(module)
feature_dir = module_dir + "/" + layer
feature_filename = "idds.npy"

if not os.path.isdir(module_dir):
    os.mkdir(module_dir)

if not os.path.isdir(feature_dir):
    os.mkdir(feature_dir)


print("getting features")
id_ds_feats = np.array([])
batch_size = 16
for batch in tqdm(range(0, id_ds_images.shape[0], batch_size)):
    image_batch = id_ds_images[batch : batch + batch_size]  # (batch, 3, 416, 416)
    image_batch = torch.tensor(image_batch).float()
    prediction_batch = model(image_batch.cuda())

    inter_features_batch = features["feats"]  # (batch, 32, 416, 416)
    inter_features_batch = np.array(inter_features_batch.cpu())
    inter_features_mean_batch = batch_image_mean(inter_features_batch)  # (batch, 32)
    if id_ds_feats.shape[0] > 0:
        id_ds_feats = np.append(id_ds_feats, inter_features_mean_batch, 0)
    else:
        id_ds_feats = inter_features_mean_batch

    print(id_ds_feats.shape)
    del image_batch, prediction_batch, inter_features_mean_batch, inter_features_batch
    gc.collect()
    torch.cuda.empty_cache()

print("Saving feats")
np.save(os.path.join(feature_dir, feature_filename), id_ds_feats)
# np.save("tds_features/tds_feats_0_conv_0_4.npy", td_ds_feats)

print("DONE!")
