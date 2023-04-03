mport numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import copy
from sklearn.metrics import classification_report
import random
import numpy as np
from numpy import vstack
from numpy import argmax
from PIL import Image


# PASO 1: Creamos una clase para cargar los datos de entrenamiento y validación
class VehicleDataset(Dataset):
    def __init__(self, image_folder, labels_folder, transform = None):
        self.image_folder = image_folder
        self.labels_folder = labels_folder
        self.transform = transform

        # Lee la lista de imágenes y etiquetas
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(labels_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Lee la imagen
        image_path = os.path.join(self.image_folder, self.image_files[index])
        image = Image.open(image_path).convert("RGB")

        # Lee la etiqueta
        label_path = os.path.join(self.labels_folder, self.label_files[index])
        with open(label_path, "r") as f:
            lineas = f.readlines()

        # Convierte las etiquetas a anotaciones pytorch
        boxes = []
        labels = []
        for linea in lineas:
            clase, x_min, y_min, ancho, alto = [float(x) for x in linea.split()]
            x_max = x_min + ancho
            y_max = y_min + alto
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(clase)

        # Convierte las listas de anotaciones en tensores
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Aplique la transformación a la imagen
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, boxes, labels






# Testear la clase VehicleDataset
ruta_imagenes = "Dataset/train/images"
ruta_anotaciones = "Dataset/train/labels"
dataset = VehicleDataset(ruta_imagenes, ruta_anotaciones)
print(dataset[0])


# Paso 2: Crear clase para definir el modelo

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Define las capas del modelo
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64*16*16, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=7)

    def forward(self, x):
        # Implementa la propagación hacia adelante del modelo
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = x.view(-1, 64*16*16)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
