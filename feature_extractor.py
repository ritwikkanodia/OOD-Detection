from models import *
import os
import torch
import numpy as np
import gc
from tqdm import tqdm
import numpy as np
from helper import batch_image_mean, load_model
import time


def extract_features_layer(model, data_type, batch_size):
    # layer = "conv_6"

    # register the hook
    features = {}

    def get_features(name):
        def hook(model, input, output):
            # print("output.shape: ", len(output))
            features[name] = output[0].detach()

        return hook

    modules = [
        # 0,
        # 1,
        # 2,
        # 3,
        # 5,
        # 6,
        # 7,
        # 9,
        # 10,
        # 12,
        # 13,
        # 14,
        # 16,
        # 17,
        # 19,
        # 20,
        # 22,
        # 23,
        # 25,
        # 26,
        # 28,
        # 29,
        # 31,
        # 32,
        # 34,
        # 35,
        # 37,
        # 38,
        # 39,
        # 41,
        # 42,
        # 44,
        # 45,
        # 47,
        # 48,
        # 50,
        # 51,
        # 53,
        # 54,
        # 56,
        # 57,
        # 59,
        # 60,
        # 62,
        # 63,
        # 64,
        # 66,
        # 67,
        # 69,
        # 70,
        # 72,
        # 73,
        # 75,
        # 76,
        # 77,
        # 78,
        # 79,
        # 80,
        # 81,
        82,
        # 84,
        # 87,
        # 88,
        # 89,
        # 90,
        # 91,
        # 92,
        # 93,
        94,
        # 96,
        # 99,
        # 100,
        # 101,
        # 102,
        # 103,
        # 104,
        # 105,
        106,
    ]
    yolo = [82, 94, 106]

    # model.module_list[0].conv_0.register_forward_hook(get_features("conv_0"))
    # model.module_list[1].conv_1.register_forward_hook(get_features("conv_1"))
    # model.module_list[2].conv_2.register_forward_hook(get_features("conv_2"))
    # model.module_list[3].conv_3.register_forward_hook(get_features("conv_3"))
    # model.module_list[5].conv_5.register_forward_hook(get_features("conv_5"))
    # model.module_list[6].conv_6.register_forward_hook(get_features("conv_6"))
    # model.module_list[7].conv_7.register_forward_hook(get_features("conv_7"))
    # model.module_list[9].conv_9.register_forward_hook(get_features("conv_9"))
    # model.module_list[10].conv_10.register_forward_hook(get_features("conv_10"))
    # model.module_list[12].conv_12.register_forward_hook(get_features("conv_12"))
    # model.module_list[13].conv_13.register_forward_hook(get_features("conv_13"))
    # model.module_list[14].conv_14.register_forward_hook(get_features("conv_14"))
    # model.module_list[16].conv_16.register_forward_hook(get_features("conv_16"))
    # model.module_list[17].conv_17.register_forward_hook(get_features("conv_17"))
    # model.module_list[19].conv_19.register_forward_hook(get_features("conv_19"))
    # model.module_list[20].conv_20.register_forward_hook(get_features("conv_20"))
    # model.module_list[22].conv_22.register_forward_hook(get_features("conv_22"))
    # model.module_list[23].conv_23.register_forward_hook(get_features("conv_23"))
    # model.module_list[25].conv_25.register_forward_hook(get_features("conv_25"))
    # model.module_list[26].conv_26.register_forward_hook(get_features("conv_26"))
    # model.module_list[28].conv_28.register_forward_hook(get_features("conv_28"))
    # model.module_list[29].conv_29.register_forward_hook(get_features("conv_29"))
    # model.module_list[31].conv_31.register_forward_hook(get_features("conv_31"))
    # model.module_list[32].conv_32.register_forward_hook(get_features("conv_32"))
    # model.module_list[34].conv_34.register_forward_hook(get_features("conv_34"))
    # model.module_list[35].conv_35.register_forward_hook(get_features("conv_35"))
    # model.module_list[37].conv_37.register_forward_hook(get_features("conv_37"))
    # model.module_list[38].conv_38.register_forward_hook(get_features("conv_38"))
    # model.module_list[39].conv_39.register_forward_hook(get_features("conv_39"))
    # model.module_list[41].conv_41.register_forward_hook(get_features("conv_41"))
    # model.module_list[42].conv_42.register_forward_hook(get_features("conv_42"))
    # model.module_list[44].conv_44.register_forward_hook(get_features("conv_44"))
    # model.module_list[45].conv_45.register_forward_hook(get_features("conv_45"))
    # model.module_list[47].conv_47.register_forward_hook(get_features("conv_47"))
    # model.module_list[48].conv_48.register_forward_hook(get_features("conv_48"))
    # model.module_list[50].conv_50.register_forward_hook(get_features("conv_50"))
    # model.module_list[51].conv_51.register_forward_hook(get_features("conv_51"))
    # model.module_list[53].conv_53.register_forward_hook(get_features("conv_53"))
    # model.module_list[54].conv_54.register_forward_hook(get_features("conv_54"))
    # model.module_list[56].conv_56.register_forward_hook(get_features("conv_56"))
    # model.module_list[57].conv_57.register_forward_hook(get_features("conv_57"))
    # model.module_list[59].conv_59.register_forward_hook(get_features("conv_59"))
    # model.module_list[60].conv_60.register_forward_hook(get_features("conv_60"))
    # model.module_list[62].conv_62.register_forward_hook(get_features("conv_62"))
    # model.module_list[63].conv_63.register_forward_hook(get_features("conv_63"))
    # model.module_list[64].conv_64.register_forward_hook(get_features("conv_64"))
    # model.module_list[66].conv_66.register_forward_hook(get_features("conv_66"))
    # model.module_list[67].conv_67.register_forward_hook(get_features("conv_67"))
    # model.module_list[69].conv_69.register_forward_hook(get_features("conv_69"))
    # model.module_list[70].conv_70.register_forward_hook(get_features("conv_70"))
    # model.module_list[72].conv_72.register_forward_hook(get_features("conv_72"))
    # model.module_list[73].conv_73.register_forward_hook(get_features("conv_73"))
    # model.module_list[75].conv_75.register_forward_hook(get_features("conv_75"))
    # model.module_list[76].conv_76.register_forward_hook(get_features("conv_76"))
    # model.module_list[77].conv_77.register_forward_hook(get_features("conv_77"))
    # model.module_list[78].conv_78.register_forward_hook(get_features("conv_78"))
    # model.module_list[79].conv_79.register_forward_hook(get_features("conv_79"))
    # model.module_list[80].conv_80.register_forward_hook(get_features("conv_80"))
    # model.module_list[81].conv_81.register_forward_hook(get_features("conv_81"))
    model.module_list[82].yolo_82.register_forward_hook(get_features("yolo_82"))
    # model.module_list[84].conv_84.register_forward_hook(get_features("conv_84"))
    # model.module_list[87].conv_87.register_forward_hook(get_features("conv_87"))
    # model.module_list[88].conv_88.register_forward_hook(get_features("conv_88"))
    # model.module_list[89].conv_89.register_forward_hook(get_features("conv_89"))
    # model.module_list[90].conv_90.register_forward_hook(get_features("conv_90"))
    # model.module_list[91].conv_91.register_forward_hook(get_features("conv_91"))
    # model.module_list[92].conv_92.register_forward_hook(get_features("conv_92"))
    # model.module_list[93].conv_93.register_forward_hook(get_features("conv_93"))
    model.module_list[94].yolo_94.register_forward_hook(get_features("yolo_94"))
    # model.module_list[96].conv_96.register_forward_hook(get_features("conv_96"))
    # model.module_list[99].conv_99.register_forward_hook(get_features("conv_99"))
    # model.module_list[100].conv_100.register_forward_hook(get_features("conv_100"))
    # model.module_list[101].conv_101.register_forward_hook(get_features("conv_101"))
    # model.module_list[102].conv_102.register_forward_hook(get_features("conv_102"))
    # model.module_list[103].conv_103.register_forward_hook(get_features("conv_103"))
    # model.module_list[104].conv_104.register_forward_hook(get_features("conv_104"))
    # model.module_list[105].conv_105.register_forward_hook(get_features("conv_105"))
    model.module_list[106].yolo_106.register_forward_hook(get_features("yolo_106"))

    print("hook registered")

    # load images
    images_dir = (
        "/home/FYP/ritwik002/main/"
        + data_type
        + "/"
        + data_type
        + "_images/"
        + data_type
        + "_cropped_resized_images.npy"
    )
    ds_images = np.load(images_dir)
    if ds_images.shape[1] != 3:
        ds_images = ds_images.reshape(
            (
                ds_images.shape[0],
                ds_images.shape[3],
                ds_images.shape[2],
                ds_images.shape[1],
            )
        )

    print("loaded images from: ", images_dir, " with shape ", ds_images.shape)

    for module in tqdm(modules):
        layer = "yolo_" + str(module)
        # layer = "conv_" + str(module)

        print(
            "Extracting features for module ",
            module,
            " and layer ",
            layer,
            " for ",
            data_type,
        )

        # print(module)

        # register paths
        module_dir = (
            "/home/FYP/ritwik002/main/"
            + data_type
            + "/"
            + data_type
            + "_features_cropped/module_"
            + str(module)
        )

        feature_dir = module_dir + "/" + layer
        feature_filename = data_type + "_features.npy"
        print("Storing features at: ", os.path.join(feature_dir, feature_filename))
        if not os.path.isdir(module_dir):
            os.mkdir(module_dir)

        if not os.path.isdir(feature_dir):
            os.mkdir(feature_dir)

        # print("feature_dir: ", feature_dir)
        # print("feature_filename: ", feature_filename)

        # extract features
        # print("Extracting feats")
        ds_feats = np.array([])

        for batch in range(0, ds_images.shape[0], batch_size):
            image_batch = ds_images[batch : batch + batch_size]  # (batch, 3, 416, 416)
            image_batch = torch.tensor(image_batch).float()
            prediction_batch = model(image_batch.cuda())

            inter_features_batch = features[layer]  # (batch, 32, 416, 416)
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

        # saving feats

        print("Saving feats")
        print(ds_feats.shape)
        np.save(os.path.join(feature_dir, feature_filename), ds_feats)
        del ds_feats
        print("feats saved")
        print()


config_dir = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
pretrained_weights_dir = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
model = load_model(config_dir, pretrained_weights_dir)
start_time = time.time()
extract_features_layer(model, "oodds", 32)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
extract_features_layer(model, "idds", 32)
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
extract_features_layer(model, "tds", 32)
print("--- %s seconds ---" % (time.time() - start_time))

