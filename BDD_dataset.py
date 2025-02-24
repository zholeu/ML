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

random.seed(1) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]

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


if __name__ == "__main__":
    LR = 0.001
    num_classes = 11
    batch_size = 1
    start_epoch, max_epoch = 0, 30
    base_dir = os.path.join(BASE_DIR, "data", "bdd100k")
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    train_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='train', label_list=BDD_INSTANCE_CATEGORY_NAMES)
    val_set = bddDataset(data_dir=base_dir, transforms=train_transform, flag='val', label_list=BDD_INSTANCE_CATEGORY_NAMES)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) # replace the pre-trained head with a new one

    print(device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(start_epoch, max_epoch):

        model.train()
        for iter, (images, targets) in enumerate(train_loader):

            # if iter > 100:
            #     break

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            train_loss_dict = model(images, targets)  # images is list; targets is [ dict["boxes":**, "labels":**], dict[] ]
            losses = sum(loss for loss in train_loss_dict.values())

            # if iter % 300 == 0:
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch, max_epoch, iter + 1, len(train_loader), losses.item()))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

        torch.save(model, './models/{}_model.pth.tar'.format(epoch+1))

        # # # val
    #     # model.eval()
    #     total_loss = list()
    #     best_losses = 10000  # val
    #
    #     for iter, (images, targets) in enumerate(val_loader):
    #         if iter > 100:
    #             break
    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #         val_loss_dict = model(images, targets) # images is list; targets is [ dict["boxes":**, "labels":**], dict[] ]
    #         val_losses = sum(loss for loss in val_loss_dict.values())
    #
    #         total_loss.append(val_losses)
    #
    #     val_losse = sum(total_loss) / len(total_loss)
    #     print("val:Epoch[{:0>3}/{:0>3}] Loss: {:.4f} ".format(epoch, max_epoch, val_losse))
    #     if val_losse < best_losses:
    #         best_losses = val_losse
    #         bestmodel_num = epoch + 1
    #         torch.save(model, './best_model.pth.tar')
    #
    # print('best_model_epoch:{}'.format(bestmodel_num))

    # test
    model.eval()
    vis_num = 3
    vis_dir = os.path.join(BASE_DIR, "data", "bdd100k", "images", '100k', 'test')
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

        vis_bbox(input_image, output_dict, BDD_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_thres=0.5)  
