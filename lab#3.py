import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.utils.prune as prune
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from choreographer.pipe import NumpyEncoder
from sklearn.preprocessing import LabelEncoder
import os
import time
import torch.nn as nn
import torch
import random
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
from sympy import false, true
from torch.utils.tensorboard import SummaryWriter
import json
from torchsummary import summary

from BDD_dataset import bddDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torch.onnx
import onnx
import onnxruntime as ort

random.seed(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR2 = "C:\\BDD100K"
device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# classes_BDD
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]


def vis_bbox(img, output, classes, max_vis=40, prob_thres=0.4):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    out_boxes = output["boxes"].cpu()
    out_scores = output["scores"].cpu()
    out_labels = output["labels"].cpu()

    num_boxes = out_boxes.shape[0]
    for idx in range(0, min(num_boxes, max_vis)):

        score = out_scores[idx].numpy()
        bbox = out_boxes[idx].numpy()
        class_name = classes[out_labels[idx]]

        if score < prob_thres:
            continue

        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                   edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.show()
    plt.close()


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# Function to run inference with the ONNX model
def run_inference_with_onnx(model_path, input_data):
    # Create an ONNX runtime inference session
    session = ort.InferenceSession(model_path)

    # Prepare the input (make sure it matches the input format expected by the ONNX model)
    # The input should be a numpy array of the same shape that was used during training
    # In this example, we use [1, 3, 800, 1200] as the input shape, assuming RGB images.
    # You can change this shape depending on your model input size.
    input_data = np.array(input_data, dtype=np.float32)  # Convert input to numpy array

    # Run inference (assuming 'input' is the name of the input tensor in the ONNX model)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)  # Run the model, get all outputs

    # Post-processing (e.g., extracting bounding boxes, labels, etc.)
    return outputs

def main():
    # config
    LR = 0.001
    num_classes = 11
    batch_size = 4
    start_epoch, max_epoch = 0, 10
    base_dir = os.path.join(BASE_DIR2)
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    writer = SummaryWriter("runs/experiment_1")
    torch.cuda.empty_cache()

    # step 1: data
    train_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    val_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='val', label_list=BDD_INSTANCE_CATEGORY_NAMES)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=true)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=true)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one

    print(device)
    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    training_losses = []
    losses_per_epoch = []
    accuracy_per_epoch = []
    losses_per_epoch = []
    accuracy_per_epoch = []
    precision_per_epoch = []
    recall_per_epoch = []
    f1_per_epoch = []
    map_per_epoch = []

    prune_amount = 0.2
    prune_interval = 1
    map_metric = MeanAveragePrecision(iou_type="bbox").to(device)
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives

    for epoch in range(start_epoch, max_epoch):
        model.train()
        total_loss = 0.0
        num_batches = 0

        total_correct = 0
        total_instances = 0
        try:
            for iter, (images, targets) in enumerate(train_loader):
                if iter > 100:
                    break
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                train_loss_dict = model(images, targets)  # images is list; targets is [ dict["boxes":**, "labels":**], dict[] ]
                losses = sum(loss for loss in train_loss_dict.values())
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(epoch, max_epoch, iter + 1, len(train_loader), losses.item()))
                # Calculate accuracy (IoU-based or simply matching boxes and labels)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                total_loss += losses.item()
                num_batches += 1

                # Get Predictions Separately
                model.eval()
                with torch.no_grad():
                    predictions = model(images)

                model.train()
                map_metric.update(predictions, targets)

                for target, prediction in zip(targets, predictions):
                    gt_boxes = target['boxes'].cpu()
                    pred_boxes = prediction['boxes'].cpu()
                    pred_scores = prediction['scores'].cpu()
                    # Filter predictions by confidence threshold (e.g., 0.5)
                    threshold = 0.5
                    keep = pred_scores > threshold
                    pred_boxes = pred_boxes[keep]

                    correct = sum(
                        1 for _ in filter(lambda iou: iou >= 0.5, [compute_iou([gt], pred_boxes) for gt in gt_boxes]))

                    total_correct += correct
                    total_instances += len(gt_boxes)

                    # Precision / Recall Calculation
                    tp = correct  # True Positives (Correctly detected objects)
                    fp = len(pred_boxes) - tp  # False Positives (Extra boxes)
                    fn = len(gt_boxes) - tp  # False Negatives (Missed objects)

                    total_tp += tp
                    total_fp += max(fp, 0)  # Avoid negatives
                    total_fn += max(fn, 0)  # Avoid negatives
        except IndexError:
                print("pass")
                pass

            # writer.add_scalar("Loss/train", losses.item(), iter + epoch * len(train_loader))
        avg_loss = total_loss / num_batches
        losses_per_epoch.append(avg_loss)  # Save the avg loss
        # Calculate accuracy for the epoch
        accuracy = total_correct / total_instances if total_instances > 0 else 0
        accuracy_per_epoch.append(accuracy)  # Save accuracy for the epoch

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        map_score = map_metric.compute()["map"]  # Extract mean AP
        # Store metrics
        precision_per_epoch.append(precision)
        recall_per_epoch.append(recall)
        f1_per_epoch.append(f1_score)
        map_per_epoch.append(map_score.item())

        writer.add_scalar("Loss/Train", avg_loss, epoch)  # Log loss to TensorBoard
        writer.add_scalar("Accuracy/Train", accuracy, epoch)  # Log accuracy to TensorBoard
        writer.add_scalar("Precision/Train", precision, epoch)
        writer.add_scalar("Recall/Train", recall, epoch)
        writer.add_scalar("F1-Score/Train", f1_score, epoch)
        writer.add_scalar("mAP/Train", map_score.item(), epoch)

        if epoch % prune_interval == 0 and epoch > 0:
            apply_l2_pruning(model, amount=prune_amount)
            print(f"Applied L2 pruning at epoch {epoch}")

        lr_scheduler.step()

        torch.save(model, 'models/{}_1model.pth.tar'.format(epoch+1))
        # Export the model to ONNX after each epoch
        onnx_model_path = 'onx/{}_1faster_rcnn_epoch.onnx'.format(epoch+1)
        export_model_to_onnx(model, onnx_model_path, device)
        torch.cuda.empty_cache()

    writer.close()

    # # config
    vis_num = 3
    vis_dir = os.path.join(BASE_DIR2, "images", '100k', 'test')
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    random.shuffle(img_names)
    preprocess = transforms.Compose([transforms.ToTensor(), ])

    for i in range(0, vis_num):

        path_img = os.path.join(vis_dir, img_names[i])
        input_image = Image.open(path_img).convert("RGB")
        img_chw = preprocess(input_image)

        if torch.cuda.is_available():
            img_chw = img_chw.to('cuda')
            model.to('cuda')

        input_list = [img_chw]
        with torch.no_grad():
            tic = time.time()
            print("input img tensor shape:{}".format(input_list[0].shape))
            output_list = model(input_list)
            output_dict = output_list[0]
            print("pass: {:.3f}s".format(time.time() - tic))

        vis_bbox(input_image, output_dict, BDD_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_thres=0.5)  # for 2 epoch for nms

    # config
    vis_num = 3
    vis_dir = os.path.join(BASE_DIR2, "images", '100k', 'test')
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    random.shuffle(img_names)
    onnx_model_path = "onx/2_1faster_rcnn_epoch.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path)
    
    for i in range(0, vis_num):
        path_img = os.path.join(vis_dir, img_names[i])
        input_image = Image.open(path_img).convert("RGB")
        img_np = preprocess2(input_image)
        print(f" Processed input image shape: {img_np.shape}")
        print(" Running ONNX inference...")
        inputs = {onnx_session.get_inputs()[0].name: img_np}
        with torch.no_grad():
            tic = time.time()
            output_list = onnx_session.run(None, inputs)
            print(f" Inference time: {time.time() - tic:.3f}s")
        boxes, labels, scores = output_list
        if boxes.shape[0] == 0:
            print("⚠️ No objects detected in the image.")
            continue
        output_dict = {
            "boxes": torch.tensor(boxes, device=device),
            "labels": torch.tensor(labels, dtype=torch.int64, device=device),
            "scores": torch.tensor(scores, device=device)
        }
        vis_bbox(input_image, output_dict, BDD_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_thres=0.5)


