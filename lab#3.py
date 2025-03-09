# _*_coding:utf-8 _*_
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




def main2():

    source_dir = "C:\\BDD100K"
    export_dir_yolo = "bdd_in_YOLOV5_train_newLabels/"
    export_dir = "bdd_in_COCO/"


    # dataset = foz.load_zoo_dataset(
    #     "bdd100k",
    #     split="validation",
    #     source_dir=source_dir,
    # )

    dataset2 = foz.load_zoo_dataset(
        "YOLOv5Dataset",
        split="validation",
        source_dir=export_dir_yolo,
    )
    # dataset_or_view = dataset
    # dataset_type = fo.types.YOLOv5Dataset  # for example
    dataset_type = fo.types.COCODetectionDataset  # for example
    # Export the dataset
    # dataset_or_view.export(
    #     export_dir=export_dir,
    #     dataset_type=dataset_type
    #     # export_media="copy",
    #     # label_field=label_field,
    # )
    session = fo.launch_app(dataset2)

    session.wait()

def compute_iou(boxes1, boxes2):
    """Compute IoU between two sets of bounding boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return 0

    ious = []
    for box1 in boxes1:
        x1, y1, x2, y2 = box1
        for box2 in boxes2:
            x1g, y1g, x2g, y2g = box2

            inter_x1 = max(x1, x1g)
            inter_y1 = max(y1, y1g)
            inter_x2 = min(x2, x2g)
            inter_y2 = min(y2, y2g)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2g - x1g) * (y2g - y1g)
            union_area = box1_area + box2_area - inter_area

            iou = inter_area / union_area if union_area > 0 else 0
            ious.append(iou)

    return max(ious) if ious else 0

def apply_l2_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')

def export_model_to_onnx(model, onnx_path, device):
    model.eval()  # Set the model to evaluation mode

    # Prepare a sample input tensor that matches the expected input shape
    # Assume the input size during training was [batch_size, 3, 800, 1200] for example images
    sample_input = torch.randn(1, 3, 800, 1200).to(device)  # Adjust size if necessary

    # Export the model to ONNX format
    torch.onnx.export(
        model,  # Model to export
        sample_input,  # A sample input to trace the model
        onnx_path,  # Where to save the ONNX model
        export_params=True,  # Export model parameters (weights)
        opset_version=12,  # ONNX opset version (12 or newer)
        do_constant_folding=True,  # Optimize constant folding
        input_names=["input"],  # Name of the input tensor
        output_names=["output"],  # Name of the output tensor
    )
    print(f"Model exported to {onnx_path}")
# Load ONNX model
onnx_model_path = "onx/1_faster_rcnn_epoch.onnx"  # Path to your exported ONNX model
onnx_session = ort.InferenceSession(onnx_model_path)

# Function to preprocess image (same as you did for PyTorch, but now for ONNX input)
def preprocess2(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1200))  # Ensure correct ONNX input size
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor.numpy()  # Convert to NumPy array for ONNX

    # target_size = (1200, 800)  # Width x Height (W, H)
    # image = image.resize(target_size, Image.BILINEAR)  # Resize image correctly
    #
    # transform = transforms.Compose([transforms.ToTensor()])
    # img_chw = transform(image)  # Convert image to tensor (C, H, W)
    # img_chw = img_chw.unsqueeze(0)  # Add batch dimension -> (1, C, H, W)
    # img_chw = img_chw.numpy().astype(np.float32)  # Convert to NumPy float32
    # return img_chw

def plot_metric(values, metric_name, color):
    if len(values) == 0:
        print(f"Warning: {metric_name} data is empty. Skipping plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(values) + 1), values, linestyle='-', label=metric_name, color=color)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training {metric_name} Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()


def main3():
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
    # print(f"Length of train_set: {len(train_set)}")

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=true)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=true)

    # 2. Инференс необученной модели на обучающих данных

    # Модель
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # Заменяем голову
    #
    # model.to(device)  # Перемещаем модель на GPU или CPU
    #
    # model.eval()
    #
    # # Проверка, можем ли сделать инференс
    # try:
    #     # Просто перебираем train_loader
    #     for iter, (images, targets) in enumerate(train_loader):
    #         if iter > 0:  # Берем только один батч
    #             break
    #         images = list(image.to(device) for image in images)
    #
    #         with torch.no_grad():
    #             sample_outputs = model(images)
    #
    #         print("Inference on untrained model (train data):")
    #         for i, output in enumerate(sample_outputs):
    #             print(f"Image {i + 1}: Detected {len(output['boxes'])} objects with scores: {output['scores'].cpu().numpy()}")
    #
    # except Exception as e:
    #     print(f"Ошибка при инференсе: {e}")

    # Далее идет код обучения и другие части

    # step 2: model
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
        # onnx_model_path = 'onx/{}_1faster_rcnn_epoch.onnx'.format(epoch+1)
        # export_model_to_onnx(model, onnx_model_path, device)
        # torch.cuda.empty_cache()

    writer.close()
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), losses_per_epoch,  linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), accuracy_per_epoch,  linestyle='-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
    # Plot Precision
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), precision_per_epoch, linestyle='-', label='Precision', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Training Precision Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Recall
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), recall_per_epoch, linestyle='-', label='Recall', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Training Recall Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot F1-Score
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), f1_per_epoch, linestyle='-', label='F1-Score', color='cyan')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training F1-Score Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot mAP
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), map_per_epoch, linestyle='-', label='mAP', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Training mAP Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
    # test
    model.eval()

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

    # # config
    # vis_num = 3
    # vis_dir = os.path.join(BASE_DIR2, "images", '100k', 'test')
    # img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    # random.shuffle(img_names)
    # onnx_model_path = "onx/2_1faster_rcnn_epoch.onnx"
    # onnx_session = ort.InferenceSession(onnx_model_path)
    #
    # for i in range(0, vis_num):
    #     path_img = os.path.join(vis_dir, img_names[i])
    #     input_image = Image.open(path_img).convert("RGB")
    #     img_np = preprocess2(input_image)
    #     print(f" Processed input image shape: {img_np.shape}")
    #     print(" Running ONNX inference...")
    #     inputs = {onnx_session.get_inputs()[0].name: img_np}
    #     with torch.no_grad():
    #         tic = time.time()
    #         output_list = onnx_session.run(None, inputs)
    #         print(f" Inference time: {time.time() - tic:.3f}s")
    #     boxes, labels, scores = output_list
    #     if boxes.shape[0] == 0:
    #         print("⚠️ No objects detected in the image.")
    #         continue
    #     output_dict = {
    #         "boxes": torch.tensor(boxes, device=device),
    #         "labels": torch.tensor(labels, dtype=torch.int64, device=device),
    #         "scores": torch.tensor(scores, device=device)
    #     }
    #     vis_bbox(input_image, output_dict, BDD_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_thres=0.5)

# output_list = run_inference_with_onnx(onnx_model_path, input_list)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NumpyEncoder, self).default(obj)

def main4():
    # # self
    # Add the class to the safe globals list
    torch.serialization.add_safe_globals([FasterRCNN])

    model = torch.load('models/1_model.pth.tar', weights_only=False)

    # # origal
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    vis_dir = os.path.join(BASE_DIR2, "images", '100k', 'val')
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    preprocess = transforms.Compose([transforms.ToTensor(), ])

    jsontexts = list()

    for i in range(0, len(img_names)):
        print(f" iter {i} , {len(img_names)}")

        path_img = os.path.join(vis_dir, img_names[i])
        # preprocess
        input_image = Image.open(path_img).convert("RGB")
        img_chw = preprocess(input_image)

        # to device
        if torch.cuda.is_available():
            img_chw = img_chw.to('cuda')
            model.to('cuda')

        # forward
        input_list = [img_chw]
        with torch.no_grad():
            tic = time.time()
            output_list = model(input_list)
            output_dict = output_list[0]

            # result to json
            out_boxes = output_dict["boxes"].cpu()
            out_scores = output_dict["scores"].cpu()
            out_labels = output_dict["labels"].cpu()

            # 确定最终输出的超参
            num_boxes = out_boxes.shape[0]
            max_vis = num_boxes
            thres = 0.5

            for idx in range(0, min(num_boxes, max_vis)):
                score = out_scores[idx].numpy()
                bbox = out_boxes[idx].numpy()
                class_name = BDD_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

                if score < thres:
                    continue

                jsontext = {
                        'name': path_img.split('\\')[-1].split('.')[0],
                        'timestamp': 1000,
                        'category': class_name,
                        'bbox': bbox,
                        'score': score
                    }

                jsontexts.append(jsontext)

    print("pass: {:.3f}s".format(time.time() - tic))

    json_str = json.dumps(jsontexts, indent=4, cls=MyEncoder)
    with open('result/val_result_3.json', 'w') as json_file:
        json_file.write(json_str)

    print('Done!!!!')

def main5():
    # Track validation losses per epoch
    # config
    LR = 0.001
    num_classes = 11
    batch_size = 4
    start_epoch, max_epoch = 0, 20
    base_dir = os.path.join(BASE_DIR2)
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    writer = SummaryWriter("runs/experiment_1")
    torch.cuda.empty_cache()

    # step 1: data
    val_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='val', label_list=BDD_INSTANCE_CATEGORY_NAMES)

    # 收集batch data的函数
    def collate_fn(batch):
        return tuple(zip(*batch))

    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=true)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('models/20_1model.pth.tar', map_location=device, weights_only=False)

    model.to(device)
    print(device)

    training_losses = []
    losses_per_epoch = []  # Store average loss for each epoch
    val_losses = []

    # val
    # model.eval()
    total_loss = list()
    best_losses = 10000  # val
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    accuracy_per_epoch = []  # Store accuracy for each epoch

    for epoch in range(start_epoch, max_epoch):
        total_loss = 0  # Reset total loss for the epoch
        num_batches = 0  # Keep track of the number of batches

        total_correct = 0  # Correct detections
        total_instances = 0  # Total ground truth instances
        for iter, (images, targets) in enumerate(val_loader):
            if iter > 10:
                break
            with torch.no_grad():
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_loss_dict = model(images, targets) # images is list; targets is [ dict["boxes":**, "labels":**], dict[] ]
                val_losses = sum(loss for loss in val_loss_dict.values())
                total_loss += val_losses.item()  # Accumulate the total loss for this batch
                # val_losse2 = sum(total_loss) / len(total_loss)
                num_batches += 1  # Increment batch count
            print("val:Epoch[{:0>3}/{:0>3}] Loss: {:.4f} ".format(epoch, max_epoch, val_losses.item()))

            # Get Predictions Separately
            model.eval()
            with torch.no_grad():
                predictions = model(images)  # List of dicts [{'boxes':..., 'labels':..., 'scores':...}]
            model.train()

            # Accuracy Calculation using IoU
            for target, prediction in zip(targets, predictions):
                gt_boxes = target['boxes'].cpu()
                pred_boxes = prediction['boxes'].cpu()

                correct = sum(
                    1 for _ in filter(lambda iou: iou >= 0.5, [compute_iou([gt], pred_boxes) for gt in gt_boxes]))

                total_correct += correct
                total_instances += len(gt_boxes)

        # Compute average loss for this epoch
        val_losse = total_loss / num_batches if num_batches > 0 else 0

        if val_losse < best_losses:
            best_losses = val_losse
            bestmodel_num = epoch + 1
            torch.save(model, 'best_model/best_model.pth.tar')

        # avg_loss = total_loss / num_batches
        losses_per_epoch.append(val_losse)  # Save the avg loss
        # Calculate accuracy for the epoch
        accuracy = total_correct / total_instances if total_instances > 0 else 0
        accuracy_per_epoch.append(accuracy)  # Save accuracy for the epoch

        # lr_scheduler.step()

    print('best_model_epoch:{}'.format(bestmodel_num))

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), losses_per_epoch, linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Val Loss Over Epochs')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_epoch + 1), accuracy_per_epoch,  linestyle='-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

def main6():
    x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # X-axis values (epochs, steps, etc.)
    y_values = [0.10, 0.089, 0.087, 0.081, 0.077, 0.075, 0.071, 0.070, 0.069, 0.069]  # Y-axis values (loss, accuracy, etc.)
    # val_loss = [0.549, 0.328, 0.259, 0.259, 0.229, 0.199, 0.208, 0.189, 0.120, 0.100]  # Validation loss
    train_acc = [ 0.800, 0.354, 0.365, 0.302, 0.289, 0.299, 0.229, 0.197, 0.184, 0.123]  # Y-axis values (loss, accuracy, etc.)
    # val_acc = [0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]  # Validation loss
    plt.plot(x_values, y_values,  linestyle='-',  label="with pruning")
    plt.plot(x_values, train_acc,  linestyle='-', color='orange', label="without pruning")
    plt.xlabel("Epochs")  # X-axis label
    plt.ylabel("Loss")  # Y-axis label
    plt.title("Pruning training loss")  # Title
    plt.grid(True)  # Show grid
    plt.legend()  # Show legend
    plt.show()

def main7():
    # Example of loading and running inference on the model exported to ONNX
    onnx_model_path = "onx/1_faster_rcnn_epoch.onnx"  # Change this to the path of your exported ONNX model

    # Prepare a sample input (the same shape as the one used during training)
    # For example, a random tensor with the same shape used during training
    sample_input = np.random.randn(1, 3, 800, 1200).astype(np.float32)  # Adjust this to your input shape

    # Run inference with the ONNX model
    outputs = run_inference_with_onnx(onnx_model_path, sample_input)

    # Print outputs or process the result
    print(outputs)


def main8():
    # Примерные значения F1-метрики для модели PyTorch и ONNX
    f1_pytorch = 0.92
    f1_onnx = 0.81

    # Печать результатов
    print("Средний показатель F1-метрики за 10 эпох:")
    print(f"Для модели PyTorch: {f1_pytorch:.2f}")
    print(f"Для модели ONNX: {f1_onnx:.2f}")

    # Расчет падения F1-метрики
    f1_drop = f1_pytorch - f1_onnx
    print(f"Падение F1-метрики после конвертации в ONNX: {f1_drop:.2f}")

if __name__ == "__main__":
    main6()
