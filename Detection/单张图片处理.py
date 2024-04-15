import streamlit as st
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v2, resnet50

from my_predict import process_image_2


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


# 两列 宽度1:2
col3, col4 = st.columns([1, 2])
# 第一列 logo图片 100px宽度 高度自适应
col3.image(Image.open(r'img.png'), width=150)
# 第二列 标题
col4.title('智能井盖监测系统')

col5, col6 = (st.columns(2))
file = col5.file_uploader('选择井盖图片上传')
col6.text_input('位置')
col6.text_input('上传者')
# 将 UploadedFile 对象转换为字节对象
if file is not None:
    image_bytes = file.read()
    # 调用 process_image_2 函数，并将字节对象传递给它
    result = process_image_2(image_bytes)
    # 两列
    col1, col2 = st.columns(2)
    # 第一列展示上传的图片
    col1.image(file, caption='Uploaded Image.', use_column_width=True)
    # 第二列展示处理后的图片
    col2.image(result, caption='Processed Image.', use_column_width=True)
