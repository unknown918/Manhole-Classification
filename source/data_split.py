import os
import shutil
import numpy as np
from tqdm import tqdm

# 定义数据集根目录
dataset_root = '../original'

# 定义划分比例
train_ratio = 8/13
val_ratio = 3/13
test_ratio = 2/13

# 遍历数据集根目录下的子文件夹
for folder in os.listdir(dataset_root):
    folder_path = os.path.join(dataset_root, folder)

    # 如果是文件夹而非文件
    if os.path.isdir(folder_path):
        # 获取当前文件夹下的所有文件
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # 使用随机种子进行划分
        np.random.seed(42)
        np.random.shuffle(files)

        # 计算划分点
        num_samples = len(files)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)
        num_test = num_samples - num_train - num_val

        # 划分数据集
        train_files = files[:num_train]
        val_files = files[num_train:num_train + num_val]
        test_files = files[num_train + num_val:]

        # 定义保存路径
        train_folder = os.path.join('../datasets/train', folder)
        val_folder = os.path.join('../datasets/val', folder)
        test_folder = os.path.join('../datasets/test', folder)

        # 创建保存文件的文件夹
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # 复制文件到相应的文件夹，并添加进度条
        print(f'Copying files for {folder}...')
        for file in tqdm(train_files, desc='Train'):
            shutil.copy(os.path.join(folder_path, file), os.path.join(train_folder, file))
        for file in tqdm(val_files, desc='Validation'):
            shutil.copy(os.path.join(folder_path, file), os.path.join(val_folder, file))
        for file in tqdm(test_files, desc='Test'):
            shutil.copy(os.path.join(folder_path, file), os.path.join(test_folder, file))

print('Data split and copied successfully!')
