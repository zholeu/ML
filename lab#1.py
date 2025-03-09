import os
import time
import torch.nn as nn
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
from tools.BDD_dataset import bddDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

# Список классов объектов, в нашем датасете их 11, которые модель будет детектить
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]

# Функция для визуализации предсказанных bounding box
def vis_bbox(img, output, classes, max_vis=40, prob_thres=0.4):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    
    out_boxes = output_dict["boxes"].cpu()
    out_scores = output_dict["scores"].cpu()
    out_labels = output_dict["labels"].cpu()
    
    num_boxes = out_boxes.shape[0]
    for idx in range(0, min(num_boxes, max_vis)):

        score = out_scores[idx].numpy()
        bbox = out_boxes[idx].numpy()
        class_name = classes[out_labels[idx]]

        if score < prob_thres:
            continue
        # Рисуем рамку и подписываем объект
        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                   edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.show()
    plt.close()

# Класс объединяет несколько трансформаций
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# Класс для случайного горизонтального отражения изображения
class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # Отражаем изображение по горизонтали
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]] # Корректируем координаты рамок
            target["boxes"] = bbox
        return image, target

# Класс изображения -> тензор
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image) # изображение -> формат тензора
        return image, target
        
# Подготавливаем данные (загрузчик)
def prepare_data(base_dir, train_transform, batch_size):
    train_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    val_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='val', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    return train_loader, val_loader
    
# Функция создания модели Faster R-CNN с заменой классификатора
def build_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # Меняем классификатор на новый
    
    model.to(device)
    
    return model
    
# Обучаем
def train(model, train_loader, optimizer, lr_scheduler, device, max_epoch, writer, prune_amount):
    losses_per_epoch = []
    accuracy_per_epoch = []
    
    for epoch in range(max_epoch):
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_instances = 0

        try:
            for iter, (images, targets) in enumerate(train_loader):
                if iter > 300: # Ограничение числа итераций для уменьшения времени обучения, так как датасет большой а наших выч.мощностей не всегда хватает
                    break
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                train_loss_dict = model(images, targets)
                losses = sum(loss for loss in train_loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                total_loss += losses.item()
                num_batches += 1

                with torch.no_grad():
                    predictions = model(images)

                for target, prediction in zip(targets, predictions):
                    gt_boxes = target['boxes'].cpu()
                    pred_boxes = prediction['boxes'].cpu()

                    correct = sum(
                        1 for _ in filter(lambda iou: iou >= 0.5, [compute_iou([gt], pred_boxes) for gt in gt_boxes]))

                    total_correct += correct
                    total_instances += len(gt_boxes)
# может произойти если оперативная память маленькая, поэтому для того чтобы прогресс сохранялся был добавлен данный exception
        except IndexError:
            print("pass")
            pass

        avg_loss = total_loss / num_batches
        losses_per_epoch.append(avg_loss)
        
        accuracy = total_correct / total_instances if total_instances > 0 else 0
        accuracy_per_epoch.append(accuracy)

        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Accuracy/Train", accuracy, epoch)

        # if epoch % prune_interval == 0 and epoch > 0:
            # apply_l2_pruning(model, amount=prune_amount)
            # print(f"Applied L2 pruning at epoch {epoch}")
        
        lr_scheduler.step()
    
    return losses_per_epoch, accuracy_per_epoch

def evaluate(model, val_loader, device):
    model.eval()
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            for target, prediction in zip(targets, predictions):
                gt_boxes = target['boxes'].cpu()
                pred_boxes = prediction['boxes'].cpu()

                correct = sum(
                    1 for _ in filter(lambda iou: iou >= 0.5, [compute_iou([gt], pred_boxes) for gt in gt_boxes]))

                total_correct += correct
                total_instances += len(gt_boxes)

    accuracy = total_correct / total_instances if total_instances > 0 else 0
    return accuracy
    
# Функция тестирования гипотезы
# Гиперпараметры, которые настраивались:
#     Learning Rate (lr) - скорость обучения
#     Batch Size (batch_size) - размер мини-пакета
#     Number of Epochs (num_epochs) - количество эпох
#     Pruning Amount (prune_amount) - степень pruning весов(обрезки)
#     step_size для lr_scheduler - шаг для уменьшения скорости обучения
def test_hypothesis(train_function, base_dir, device, num_classes, batch_size, num_epochs, lr, prune_amount, writer):
    
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    train_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_losses, train_accuracies = train(model, train_loader, optimizer, lr_scheduler, device, num_epochs, writer, prune_amount)

    return train_losses, train_accuracies

