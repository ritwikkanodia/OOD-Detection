import numpy as np

"""
all_images = np.load("all_images_main.npy")
all_images = all_images.reshape(
    (all_images.shape[0], all_images.shape[3], all_images.shape[1], all_images.shape[2])
)

all_images_1 = all_images[:2500]
all_images_2 = all_images[2500:5000]
all_images_3 = all_images[5000:7500]
all_images_4 = all_images[7500:]


print(all_images_1.shape)
print(all_images_2.shape)
print(all_images_3.shape)
print(all_images_4.shape)

np.save("tds_images/tds_images_1.npy", all_images_1)
print("Done")
np.save("tds_images/tds_images_2.npy", all_images_2)
print("Done")
np.save("tds_images/tds_images_3.npy", all_images_3)
print("Done")
np.save("tds_images/tds_images_4.npy", all_images_4)
print("Done")

all_images_1 = np.load("tds_images/tds_images_1.npy")
all_images_2 = np.load("tds_images/tds_images_2.npy")
all_images_3 = np.load("tds_images/tds_images_3.npy")
all_images_4 = np.load("tds_images/tds_images_4.npy")

print(all_images_1.shape)
print(all_images_2.shape)
print(all_images_3.shape)
print(all_images_4.shape)


all_images = np.append(all_images_1, all_images_2, 0)
all_images = np.append(all_images, all_images_3, 0)
all_images = np.append(all_images, all_images_4, 0)
print(all_images.shape)

all_images_1 = np.load("tds_images/tds_images_1.npy")
all_images_2 = np.load("tds_images/tds_images_2.npy")
all_images_3 = np.load("tds_images/tds_images_3.npy")
all_images_4 = np.load("tds_images/tds_images_4.npy")

print(all_images_1.shape)
print(all_images_1.shape)
print(all_images_1.shape)
print(all_images_1.shape)

all_images = np.append(all_images_1, all_images_2, all_images_3, all_images_4)
print(all_images.shape)
"""


"""
tds_images_1 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_images_resized_2500.npy"
)
tds_images_2 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_images_resized_5000.npy"
)
tds_images_3 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_images_resized_7500.npy"
)
tds_images_4 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_images_resized_10000.npy"
)
tds_images_5 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_images_resized_12500.npy"
)
tds_images_6 = np.load(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_resized_images_ex_2500.npy"
)

print(tds_images_1.shape)
print(tds_images_2.shape)
print(tds_images_3.shape)
print(tds_images_4.shape)
print(tds_images_5.shape)
print(tds_images_6.shape)


all_tds_images = np.append(tds_images_1, tds_images_2, 0)
all_tds_images = np.append(all_tds_images, tds_images_3, 0)
all_tds_images = np.append(all_tds_images, tds_images_4, 0)
all_tds_images = np.append(all_tds_images, tds_images_5, 0)
all_tds_images = np.append(all_tds_images, tds_images_6, 0)
print(all_tds_images.shape)


np.save(
    "/home/FYP/ritwik002/main/tds/tds_images/tds_cropped_resized_images.npy",
    all_tds_images,
)

print("SAVED")

del (
    tds_images_1,
    tds_images_2,
    tds_images_3,
    tds_images_4,
    tds_images_5,
    tds_images_6,
    all_tds_images,
)


oodds_images_1 = np.load(
    "/home/FYP/ritwik002/main/oodds/oodds_images/oodds_cropped_resized_images_2500.npy"
)
oodds_images_2 = np.load(
    "/home/FYP/ritwik002/main/oodds/oodds_images/oodds_cropped_resized_images_5000.npy"
)
oodds_images_3 = np.load(
    "/home/FYP/ritwik002/main/oodds/oodds_images/oodds_cropped_resized_images_7500.npy"
)

oodds_images_4 = np.load(
    "/home/FYP/ritwik002/main/oodds/oodds_images/oodds_cropped_resized_images_9000.npy"
)

print(oodds_images_1.shape)
print(oodds_images_2.shape)
print(oodds_images_3.shape)
print(oodds_images_4.shape)


all_oods_images = np.append(oodds_images_1, oodds_images_2, 0)
all_oods_images = np.append(all_oods_images, oodds_images_3, 0)
all_oods_images = np.append(all_oods_images, oodds_images_4, 0)
print(all_oods_images.shape)

np.save(
    "/home/FYP/ritwik002/main/oodds/oodds_images/oodds_cropped_resized_images.npy",
    all_oods_images,
)

print("SAVED")

del oodds_images_1, oodds_images_2, oodds_images_3, oodds_images_4, all_oods_images


idds_images_1 = np.load(
    "/home/FYP/ritwik002/main/idds/idds_images/idds_cropped_resized_images_2500.npy"
)
idds_images_2 = np.load(
    "/home/FYP/ritwik002/main/idds/idds_images/idds_cropped_resized_images_5000.npy"
)
idds_images_3 = np.load(
    "/home/FYP/ritwik002/main/idds/idds_images/idds_cropped_resized_images_7500.npy"
)
idds_images_4 = np.load(
    "/home/FYP/ritwik002/main/idds/idds_images/idds_cropped_resized_images_9000.npy"
)

print(idds_images_1.shape)
print(idds_images_2.shape)
print(idds_images_3.shape)
print(idds_images_4.shape)


all_idds_images = np.append(idds_images_1, idds_images_2, 0)
all_idds_images = np.append(all_idds_images, idds_images_3, 0)
all_idds_images = np.append(all_idds_images, idds_images_4, 0)
print(all_idds_images.shape)

np.save(
    "/home/FYP/ritwik002/main/idds/idds_images/idds_cropped_resized_images.npy",
    all_idds_images,
)

print("SAVED")

del idds_images_1, idds_images_2, idds_images_3, idds_images_4, all_idds_images

"""


# img_arr = cv2.imread(images_dir + "/" + image_name)

# tds_images = np.load("/home/FYP/ritwik002/main/tds/tds_images/tds_masked_images.npy")
# print(tds_images.shape)

# tds_images.reshape((10000, 3, 416, 416))

# np.save("/home/FYP/ritwik002/main/tds/tds_images/tds_masked_images.npy", tds_images)
# print("Done")

# from helper import cropAndResize
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import math

"""
# given the labels after detction on Kitti val, convert to array
root = "/home/FYP/ritwik002/data"
# root = "data"
kitti_val_images = os.listdir(root + "/kitti_val/images")[1:]
count = 0

for img_name in tqdm(kitti_val_images):
    label_dir = (
        root
        + "/kitti_val/kitti_val_detection/labels/"
        + img_name[:-1].split(".")[0]
        + ".txt"
    )  # labels generated oby running detection with yolo
    image_dir = root + "/kitti_val/images/" + img_name  # original validation images
    cropped_image_dir = (
        root + "/kitti_val/kitti_val_detection/id_ds_cropped_images/"
    )  # dir to store masked image
    # image = cv2.imread(image_dir)
    img = Image.open(image_dir)
    if count == 0:
        print("For image:")
        print(label_dir)
        print(image_dir)
        print(cropped_image_dir)
        print()
    bbox_count = 0
    with open(label_dir) as fp:
        Lines = fp.readlines()
        if len(Lines) != 0:
            for line in Lines:
                # print("for bbox ", count)
                bbox_coords = line.split(",")
                # print(bbox_coords)
                bbox_coords = [math.floor(float(x)) for x in bbox_coords]
                x1, x2, y1, y2 = (
                    bbox_coords[0],
                    bbox_coords[1],
                    bbox_coords[2],
                    bbox_coords[3],
                )

                # masking
                # masked_img = maskAndResize(image, x1, y1, x2, y2)
                cropped_img = cropAndResize(img, x1, y1, x2, y2)
                cropped_img.save(
                    cropped_image_dir + str(count) + "_" + str(bbox_count) + ".png"
                )
                # cv2.imwrite(masked_image_dir + str(count) + ".png", masked_img) # imgs are alr stored

                count += 1
                bbox_count += 1
                del img, cropped_img
"""


"""
# get oods bbox from val images
root = "/home/FYP/ritwik002/data"
# root = "data"
kitti_val_images = os.listdir(root + "/kitti_val/images")[1:]
count = 0

for img_name in tqdm(kitti_val_images):
    label_dir = (
        root + "/kitti_val/kitti_labels/" + img_name[:-1].split(".")[0] + ".txt"
    )  # Kitti labels
    image_dir = root + "/kitti_val/images/" + img_name
    cropped_image_dir = root + "/kitti_val/ood_ds_cropped_images/"
    # image = cv2.imread(image_dir)
    img = Image.open(image_dir)
    if count == 0:
        print("For image:")
        print(label_dir)
        print(image_dir)
        print(cropped_image_dir)
        print()

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
            cropped_img = cropAndResize(img, x1, y1, x2, y2)
            cropped_img.save(
                cropped_image_dir
                + img_name
                + "_"
                + label
                + "_"
                + str(bbox_count)
                + ".png"
            )
            count += 1
            bbox_count += 1
"""
oodds_feat_0 = np.load(
    "/Users/ritwikkanodia/Desktop/NTU/NTU 4.2/FYP/main/features/oodds/oodds_features/module_0/conv_0/oodds_features.npy"
)

oodds_feat_1 = np.load(
    "/Users/ritwikkanodia/Desktop/NTU/NTU 4.2/FYP/main/features/oodds/oodds_features_cropped/module_0/conv_0/oodds_features.npy"
)
print(oodds_feat_0.shape)
print(oodds_feat_1.shape)
