from ultralytics import YOLO

# 模型的路径 .pt
model_path = 'runs/detect/train4/weights/best.pt'
# 要预测的图片的相对路径 可以是文件夹或者单张图片
input_file = "datasets/well/test"

# 结果保存在runs/detect/predict/labels文件夹下

model = YOLO(model_path, task='detect')
results = model(source=input_file, save=True, save_txt=True,iou=0.5,conf=0.4)
