from models import *

import torch
import os
import numpy as np
from tqdm import tqdm
from utils.utils import *
from utils.datasets import *
import cv2
from torch.utils.data import DataLoader
import gc
from helper import maskAndResize, batch_image_mean


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
module = 1
layer = "conv_1"

print("FEATURE EXTRACTION FOR MODULE ", module, ", LAYER, ", layer)

features = {}


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


model.module_list[module].conv_1.register_forward_hook(get_features("feats"))

print("Hook registered for layer conv_1")

root = "/home/FYP/ritwik002/data"
# root = "data"
kitti_val_images = os.listdir(root + "/kitti_val/images")[1:]
count = 0
id_ds_feats = np.array([])
module_dir = "/home/FYP/ritwik002/main/idds/idds_features/module_" + str(module)
feature_dir = module_dir + "/" + layer
if not os.path.isdir(module_dir):
    os.mkdir(module_dir)

if not os.path.isdir(feature_dir):
    os.mkdir(feature_dir)

feature_file = "idds.npy"

for img_name in tqdm.tqdm(kitti_val_images):
    label_dir = (
        root + "/kitti_val_detection/labels/" + img_name[:-1].split(".")[0] + ".txt"
    )  # labels generated oby running detection with yolo
    image_dir = root + "/kitti_val/images/" + img_name  # original validation images
    masked_image_dir = (
        root + "/kitti_val/masked_images_resized/"
    )  # dir to store masked image
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
                # cv2.imwrite(masked_image_dir + str(count) + ".png", masked_img) # imgs are alr stored
                count += 1
                bbox_count += 1

                # aggregating all bbox in an img as a batch
                if image_batch.shape[0] > 0:
                    image_batch = np.append(
                        image_batch, masked_img.reshape((1, 3, 416, 416)), 0
                    )
                else:
                    image_batch = masked_img.reshape((1, 3, 416, 416))

                # print("image_batch.shape: ", image_batch.shape)
                del masked_img

            # prediction
            print("image_batch.shape: ", image_batch.shape)
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
            if count >= 9000:
                print("Enough images alr")
                break
print("Saving feats at: ", feature_dir)
np.save(os.path.join(feature_dir, feature_file), id_ds_feats)
print("DONE")


"""
image_folder = "/home/FYP/ritwik002/data/kitti_val/images"
model_def = "/home/FYP/ritwik002/main/yolov3-custom.cfg"
weights_path = "/home/FYP/ritwik002/main/yolov3_ckpt_48.pth"
class_path = "/home/FYP/ritwik002/main/classes.names"
conf_thres = 0.8
nms_thres = 0.4
batch_size = 1
n_cpu = 0
img_size = 416


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("/home/FYP/ritwik002/data/kitti_test_detection", exist_ok=True)
os.makedirs("/home/FYP/ritwik002/data/kitti_test_detection/labels", exist_ok=True)
os.makedirs("/home/FYP/ritwik002/data/kitti_test_detection/images", exist_ok=True)

# Set up model
model = Darknet(model_def, img_size=img_size).to(device)

if weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))

model.eval()  # Set in evaluation mode

dataloader = DataLoader(
    ImageFolder(image_folder, img_size=img_size),
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_cpu,
)

classes = load_classes(class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print("\nPerforming object detection:")
prev_time = time.time()
for batch_i, (img_paths, input_imgs) in enumerate(tqdm.tqdm(dataloader)):
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Log progress
    current_time = time.time()
    inference_time = datetime.timedelta(seconds=current_time - prev_time)
    prev_time = current_time
    print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

    # Save image and detections
    imgs.extend(img_paths)
    img_detections.extend(detections)
    sleep(0.1)

# Bounding-box colors
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

print("\nSaving images:")
# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate((zip(imgs, img_detections))):

    print("(%d) Image: '%s'" % (img_i, path))

    # Create plot
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw bounding boxes and labels of detections
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        bbox_coords = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            bbox_coords.append(
                str(round(x1.item(), 3))
                + ","
                + str(round(x2.item(), 3))
                + ","
                + str(round(y1.item(), 3))
                + ","
                + str(round(y2.item(), 3))
            )
            print(
                "\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item())
            )

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle(
                (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none"
            )
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = path.split("/")[-1].split(".")[0]
    plt.savefig(
        f"/home/FYP/ritwik002/data/kitti_test_detection/images/{filename}.png",
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close()
    # print(bbox_coords)
    with open(
        f"/home/FYP/ritwik002/data/kitti_test_detection/labels/{filename}.txt", "w"
    ) as f:
        f.write("\n".join(bbox_coords))

"""
