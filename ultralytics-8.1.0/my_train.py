import torch

from ultralytics import YOLO

# 预训练的模型
model = YOLO("./ultralytics/cfg/models/v8/yolov8.yaml")

# train
model.train(data='./datasets/well/well_data.yaml', epochs=300, patience=20, batch=-1, workers=0)
