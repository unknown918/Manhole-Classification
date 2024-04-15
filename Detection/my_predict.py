import io
import os
import shutil
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageDraw
from flask import Flask, request
from flask import send_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet50

from ultralytics import YOLO

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 数据预处理
class CustomDataset(Dataset):
    def __init__(self, xml_dir, transform=None):
        self.xml_dir = xml_dir
        self.transform = transform
        self.label_map = {'good': 0, 'broke': 1, 'lose': 2, 'uncovered': 3, 'circle': 4}
        self.data = self.parse_xml(xml_dir)

    def parse_xml(self, xml_dir):
        data = []
        for xml_file in os.listdir(xml_dir):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            path = root.find('path').text
            label = root.find('object/name').text
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)
            data.append((path, (xmin, ymin, xmax, ymax), label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, bbox, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        # 使用边界框坐标来裁剪图像
        image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        if self.transform:
            image = self.transform(image)
        return image, self.label_map[label]


# 构建模型
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.resnet = resnet50(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        mobilenet_out = self.mobilenet(x)
        resnet_out = self.resnet(x)
        return mobilenet_out, resnet_out


def test_model(model, xml_folder, transform, device):
    results = []  # 存储所有的结果
    for file_name in os.listdir(xml_folder):
        if file_name.endswith('.xml'):
            xml_file = os.path.join(xml_folder, file_name)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            path = '../{}.png'.format(root.find('folder').text)
            xmin = int(root.find('object/bndbox/xmin').text)
            ymin = int(root.find('object/bndbox/ymin').text)
            xmax = int(root.find('object/bndbox/xmax').text)
            ymax = int(root.find('object/bndbox/ymax').text)

            # Load and preprocess the test image
            image = Image.open(path).convert("RGB")
            object_image = image.crop((xmin, ymin, xmax, ymax))
            test_image = transform(object_image).unsqueeze(0)

            # Send the image through the model
            model.eval()
            with torch.no_grad():
                test_tensor = test_image.to(device)
                mobilenet_out, resnet_out = model(test_tensor)
                combined_out = mobilenet_out + resnet_out
                probabilities = F.softmax(combined_out, dim=1)
                predicted_index = probabilities.argmax(1).item()
                confidence = probabilities.max(1).values.item()
                predicted_label = {0: 'good', 1: 'broke', 2: 'lose', 3: 'uncovered', 4: 'circle'}

            # Create JSON object
            result_dict = {
                'predicted_index': predicted_label[predicted_index],
                'confidence': confidence,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }

            results.append(result_dict)  # 添加结果到列表中
    return results


def predict(input_file):
    # 模型的路径 .pt
    model_path = 'best.pt'

    # 结果保存在runs/detect/predict/labels文件夹下

    model = YOLO(model_path, task='detect')
    model(source=input_file, save=True, save_txt=True, iou=0.5, conf=0.4)


def get_json(input_path='../test.png'):
    test_dir = 'runs/detect/predict/labels'

    predict(input_path)

    saved_model_path = '../models/winwin.pth'
    trained_model = torch.load(saved_model_path, map_location=torch.device('cpu'))

    result = test_model(trained_model, test_dir, transform, device)

    shutil.rmtree("runs/detect/predict")

    return result


app = Flask(__name__)


@app.route('/process_image', methods=['POST'])
def process_image():
    # 接收图像数据
    image_data = request.data
    # 将图像数据转换成 PIL 图像对象
    image = Image.open(io.BytesIO(image_data))
    # 将图像保存到临时文件
    temp_image_path = '../test.png'
    image.save(temp_image_path)
    # 调用 get_json 函数，并得到返回的 JSON 数据
    json_data = get_json(temp_image_path)
    # 删除临时文件
    os.remove(temp_image_path)
    image = Image.open(temp_image_path)
    # 提取相关信息
    predicted_index = json_data["predicted_index"]
    confidence = json_data["confidence"]
    xmin = json_data["xmin"]
    ymin = json_data["ymin"]
    xmax = json_data["xmax"]
    ymax = json_data["ymax"]
    # 在图像上绘制边界框和标签
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin, ymin), f"{predicted_index} -  {confidence:.2f}", fill="red")
    # 将处理后的图像发送给客户端
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


def process_image_2(image_data):
    # 将图像数据转换成 PIL 图像对象
    image = Image.open(io.BytesIO(image_data))
    # 将图像保存到临时文件
    temp_image_path = '../test.png'
    image.save(temp_image_path)
    # 调用 get_json 函数，并得到返回的 JSON 数据
    json_data_list = get_json(temp_image_path)
    # 删除临时文件
    os.remove(temp_image_path)
    # 遍历预测结果列表
    for json_data in json_data_list:
        # 提取相关信息
        predicted_index = json_data["predicted_index"]
        confidence = json_data["confidence"]
        xmin = json_data["xmin"]
        ymin = json_data["ymin"]
        xmax = json_data["xmax"]
        ymax = json_data["ymax"]
        # 在图像上绘制边界框和标签
        draw = ImageDraw.Draw(image)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), f"{predicted_index} -  {confidence:.2f}", fill="red")
    return image


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    process_image()
