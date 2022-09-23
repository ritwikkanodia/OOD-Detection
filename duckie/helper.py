from ast import Pass
import numpy as np
import os
import cv2
from models import *
import torch
import gc
import pickle


def layer_feature_extractor(org_model, layer_name, ds_images):
    # register hook
    print("Extracting features for: ", layer_name)
    features = {}

    def get_features(name):
        def hook(org_model, input, output):
            # print("output.shape: ", len(output))
            features[name] = output.detach()

        return hook

    org_model.module_list[9].conv_9.register_forward_hook(get_features(layer_name))

    if ds_images.shape[1] != 3:
        ds_images = ds_images.reshape(
            (
                ds_images.shape[0],
                ds_images.shape[3],
                ds_images.shape[2],
                ds_images.shape[1],
            )
        )

    print("Data shape: ", ds_images.shape)

    ds_feats = np.array([])
    batch_size = 32
    for batch in range(0, ds_images.shape[0], batch_size):
        image_batch = ds_images[batch : batch + batch_size]  # (batch, 3, 416, 416)
        image_batch = torch.tensor(image_batch).float()
        prediction_batch = org_model(image_batch)

        inter_features_batch = features[layer_name]  # (batch, 32, 416, 416)
        inter_features_batch = np.array(inter_features_batch.cpu())
        inter_features_mean_batch = batch_image_mean(
            inter_features_batch
        )  # (batch, 32)
        if ds_feats.shape[0] > 0:
            ds_feats = np.append(ds_feats, inter_features_mean_batch, 0)
        else:
            ds_feats = inter_features_mean_batch

        # print(ds_feats.shape)
        del (
            image_batch,
            prediction_batch,
            inter_features_mean_batch,
            inter_features_batch,
        )
        gc.collect()
        torch.cuda.empty_cache()

    print("Shape of feats: ", ds_feats.shape)
    return ds_feats


def images_dir_to_arr(images_dir, num_images, save_dir=None):
    print("Coverting images in: ", images_dir, " and saving in: ", save_dir)
    images = os.listdir(images_dir)[:num_images]
    all_images = np.array([])
    count = 0
    for image_name in images:
        img_arr = cv2.imread(images_dir + "/" + image_name)
        img_arr = img_arr.reshape((1, 3, 416, 416))
        if all_images.shape[0] > 0:
            all_images = np.append(all_images, img_arr, 0)
        else:
            all_images = img_arr
        del img_arr

        count += 1
        if count % 2500 == 0:
            print("Shape of all images arr: ", all_images.shape)
            np.save(save_dir + "_" + str(count) + ".npy", all_images)
            del all_images
            all_images = np.array([])

    print("Shape of all images arr: ", all_images.shape)
    if save_dir and all_images.shape[0] != 0:
        np.save(save_dir + "_" + str(count) + ".npy", all_images)
    # return all_images


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


def load_yolo_model(config_dir, pretrained_weights_dir):
    # model_def = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
    # model_def = "yolov3-custom.cfg"
    img_size = 416
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pretrained_weights = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
    # pretrained_weights = "yolov3_ckpt_48.pth"

    model = Darknet(config_dir, img_size=img_size).to(device)
    model.load_state_dict(
        torch.load(pretrained_weights_dir, map_location=torch.device("cpu"))
    )
    print("Model Loaded")
    return model


def load_ood_detection_models(osvm_dir, ss_dir):
    osvm_model = pickle.load(open(osvm_dir, "rb"))
    ss_model = pickle.load(open(ss_dir, "rb"))
    print("OOD detection models loaded")
    return osvm_model, ss_model


def cropAndResize(img, x1, y1, x2, y2):
    img2 = img.crop((x1, y1, x2, y2))
    newsize = (416, 416)
    img2 = img2.resize(newsize)
    return img2


# images_dir_to_arr(
#     "/home/FYP/ritwik002/data/kitti_train/cropped_images_resized",
#     2500,
#     "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_resized_images_ex",
# )


def detection_single_img(img):
    pass

