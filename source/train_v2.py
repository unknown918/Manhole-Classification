import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, resnet50
from tqdm import tqdm

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
batch_size = 32
learning_rate = 0.001
num_epochs = 40

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


def plot_training_curve(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('../results/training_curve_2.png')


# 训练模型
def train_model(model, criterion, optimizer, dataloader, num_epochs=10):
    train_losses = []  # 用于保存每个epoch的训练损失值
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        # 使用tqdm包装数据加载器以添加进度条
        with tqdm(dataloader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs = inputs.to(device)
                labels = torch.tensor(labels).to(device)
                optimizer.zero_grad()
                mobilenet_out, resnet_out = model(inputs)
                loss = criterion(mobilenet_out, labels) + criterion(resnet_out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                # 更新进度条
                tepoch.set_postfix(loss=running_loss / ((tepoch.n + 1) * dataloader.batch_size))
        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)  # 保存训练损失值
        print(f'Loss: {epoch_loss:.4f}')

        path = '../models/winwin.pth'
        torch.save(model.state_dict(), path)
        print(f'Model saved to {path}')

    # 绘制训练曲线
    plot_training_curve(train_losses)

    return model


def test_model(model, xml_folder, transform):
    with open('../results.txt', 'w') as f:
        for xml_file in os.listdir(xml_folder):
            tree = ET.parse(os.path.join(xml_folder, xml_file))
            root = tree.getroot()
            path = '../test_image/{}'.format(root.find('path').text)
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
                probabilities = nn.functional.softmax(combined_out, dim=1)
                predicted_index = probabilities.argmax(1).item()
                confidence = probabilities.max(1).values.item()
                # predicted_label = {0: 'good', 1: 'broke', 2: 'lose', 3: 'uncovered', 4: 'circle'}
            result = '{}  {}  {}  {}  {}  {}\n'.format(
                root.find('path').text, predicted_index, confidence, xmin, ymin, xmax, ymax
            )
            print(result)
            f.write(result)
    f.close()


if __name__ == '__main__':
    train_dir = '../train_xmls'
    test_dir = '../test_xmls'
    # 加载数据集
    dataset = CustomDataset(xml_dir=train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 构建模型
    model = EnsembleModel(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    trained_model = train_model(model, criterion, optimizer, dataloader, num_epochs=num_epochs)

    # 测试模型
    test_model(trained_model, test_dir, transform)
