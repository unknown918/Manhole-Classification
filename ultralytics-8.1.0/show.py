import cv2
import os

# 设置图片和标签的文件夹路径
image_folder = 'datasets/well/images/test'
label_folder = 'datasets/well/labels/test'
save_folder = 'datasets/well/boxes'

# 确保保存文件夹存在
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 类别信息
classes = ['good', 'broke', 'lose', 'uncovered', 'circle']  # 根据您的类别进行修改

# 遍历图片文件夹中的所有图片
for image_name in os.listdir(image_folder):
    # if image_name.endswith('.jpg'):  # 或者'.png'等其他图片格式
    # 读取图片
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # 获取对应的标签文件
    # 如果图片格式是'.jpg'，则标签文件格式是'.txt'
    if image_name.endswith('.jpg'):
        label_name = image_name.replace('.jpg', '.txt')
        # 如果图片格式是'.png'，则标签文件格式是'.txt'
    else:
        label_name = image_name.replace('.png', '.txt')

    # 根据图片格式修改
    label_path = os.path.join(label_folder, label_name)

    # 检查标签文件是否存在
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 解析YOLO格式的标签
                class_id, x_center, y_center, width, height = map(float, line.split())

                # 转换为边界框的左上角和右下角坐标
                x1 = int((x_center - width / 2) * image.shape[1])
                y1 = int((y_center - height / 2) * image.shape[0])
                x2 = int((x_center + width / 2) * image.shape[1])
                y2 = int((y_center + height / 2) * image.shape[0])

                # 绘制边界框和类别
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, classes[int(class_id)], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)

    # 保存绘制了边界框的图片
    save_path = os.path.join(save_folder, image_name)
    cv2.imwrite(save_path, image)
