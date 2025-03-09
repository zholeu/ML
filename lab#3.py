import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import onnxruntime as ort
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from tools.BDD_dataset import bddDataset
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR2 = "C:\\BDD100K"
device = torch.device("cuda")

BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]

def vis_bbox(img, output, classes, max_vis=40, prob_thres=0.4):
    """
    Функция для визуализации bounding boxes на изображении.
    """
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

        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                   fill=False, edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2, f'{class_name} {score:.3f}', bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.show()
    plt.close()

def preprocess_onnx(image):
    """
    Функция для предобработки изображения для ONNX модели 
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((800, 1200))  # Изменение размера в соответствии с размером входных данных ONNX модели
    ])
    img_tensor = transform(image).unsqueeze(0)  # Добавление размерности для пакета
    return img_tensor.numpy()  # Преобразование в массив NumPy для ONNX

def export_model_to_onnx(model, onnx_path, device):
    """
    Экспортирует модель в формат ONNX.
    """
    model.eval()  # Устанавливаем модель в режим оценки
    sample_input = torch.randn(1, 3, 800, 1200).to(device)  # Создаем пример входных данных
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Модель экспортирована в {onnx_path}")

def run_inference_with_onnx(model_path, input_data):
    """
    Выполняет инференс с использованием ONNX модели и возвращает результаты.
    """
    session = ort.InferenceSession(model_path)
    input_data = np.array(input_data, dtype=np.float32)  # Преобразование в массив NumPy
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)  # Выполнение инференса
    return outputs

def train_and_export(model, train_loader, optimizer, lr_scheduler, device, max_epoch, writer, prune_amount):
    """
    Тренирует модель и экспортирует ее в формат ONNX после каждой эпохи.
    """
    for epoch in range(max_epoch):
        model.train()
        for iter, (images, targets) in enumerate(train_loader):
            # для теста  ограничиваем количество итераций на эпоху
            # if iter > 300:  
            #     break
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            optimizer.step()

        # Экспорт модели в формат ONNX после каждой эпохи
        onnx_model_path = f'onx/epoch_{epoch+1}_faster_rcnn.onnx'
        export_model_to_onnx(model, onnx_model_path, device)

def test_onnx():
    """
    Тестирует ONNX модель на нескольких изображениях и визуализирует результаты.
    """
    vis_num = 3
    vis_dir = os.path.join(BASE_DIR2, "images", '100k', 'test')
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    random.shuffle(img_names)
    
    onnx_model_path = "onx/2_1faster_rcnn_epoch.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path)

    for i in range(vis_num):
        path_img = os.path.join(vis_dir, img_names[i])
        input_image = Image.open(path_img).convert("RGB")
        img_np = preprocess_onnx(input_image)
        print(f"Обработанное изображение, форма: {img_np.shape}")
        print("Запуск инференса ONNX...")
        inputs = {onnx_session.get_inputs()[0].name: img_np}
        with torch.no_grad():
            tic = time.time()
            output_list = onnx_session.run(None, inputs)
            print(f"Время инференса: {time.time() - tic:.3f}s")
        boxes, labels, scores = output_list
        if boxes.shape[0] == 0:
            print("⚠️ Объекты на изображении не обнаружены.")
            continue
        
        output_dict = {
            "boxes": torch.tensor(boxes, device=device),
            "labels": torch.tensor(labels, dtype=torch.int64, device=device),
            "scores": torch.tensor(scores, device=device)
        }
        vis_bbox(input_image, output_dict, BDD_INSTANCE_CATEGORY_NAMES, max_vis=20, prob_thres=0.5)

def running_onx():
    """
    Загружает модель ONNX и выполняет инференс на случайном примере.
    """
    onnx_model_path = "onx/1_faster_rcnn_epoch.onnx"
    sample_input = np.random.randn(1, 3, 800, 1200).astype(np.float32)
    outputs = run_inference_with_onnx(onnx_model_path, sample_input)
    print(outputs)
