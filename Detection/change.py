# 写一个脚本 将文件夹下每一个txt文件的每一行的第一个字符改为0

import os


def replace_first_char(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w') as file:
                for line in lines:
                    file.write('0' + line[1:] + '\n')


replace_first_char(r'datasets/well/labels/train')  # replace with your directory
