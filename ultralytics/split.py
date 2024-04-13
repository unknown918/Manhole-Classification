import os
import shutil
import random

import os
import shutil

def move_matching_files(folder_a, folder_b, folder_c):
    # 获取文件夹a和b中的所有文件名（不包括后缀）
    files_in_a = {os.path.splitext(file)[0] for file in os.listdir(folder_a)}
    files_in_b = {os.path.splitext(file)[0] for file in os.listdir(folder_b)}

    # 找出两个文件夹中文件名相同的文件
    matching_files = files_in_a.intersection(files_in_b)

    # 如果目标文件夹不存在，则创建它
    os.makedirs(folder_c, exist_ok=True)

    # 将文件夹a中与这些文件名相同的txt文件移动到文件夹c中
    for file in os.listdir(folder_a):
        filename, ext = os.path.splitext(file)
        if filename in matching_files and ext == '.txt':
            shutil.move(os.path.join(folder_a, file), os.path.join(folder_c, file))

# 使用示例
move_matching_files('datasets/well/labels/train0', 'datasets/well/images/val', 'datasets/well/labels/val')

def split_files(src_directory, dst_directory1, dst_directory2, ratio):
    # 获取源文件夹中的所有文件
    files = os.listdir(src_directory)
    # 随机打乱文件列表
    random.shuffle(files)

    # 计算应该移动到第一个目标文件夹的文件数量
    split_index = int(len(files) * ratio)

    # 如果目标文件夹不存在，则创建它们
    os.makedirs(dst_directory1, exist_ok=True)
    os.makedirs(dst_directory2, exist_ok=True)

    # 将文件移动到目标文件夹
    for i, file in enumerate(files):
        if i < split_index:
            shutil.move(os.path.join(src_directory, file), os.path.join(dst_directory1, file))
        else:
            shutil.move(os.path.join(src_directory, file), os.path.join(dst_directory2, file))


# # 使用示例
# split_files('datasets/well/images/train0', 'datasets/well/images/train', 'datasets/well/images/vals', 0.9)
