import os
import xml.etree.ElementTree as ET
from copy import deepcopy


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root


def count_objects(xml_root):
    object_counts = {}
    for obj in xml_root.findall('object'):
        label = obj.find('name').text
        object_counts[label] = object_counts.get(label, 0) + 1
    return object_counts


def split_and_save_xmls(input_folder, output_folder, split_ratio):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 统计每个类别的对象数量
    total_object_counts = {}
    for xml_file in os.listdir(input_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(input_folder, xml_file)
            xml_root = parse_xml(xml_path)
            object_counts = count_objects(xml_root)
            for label, count in object_counts.items():
                total_object_counts[label] = total_object_counts.get(label, 0) + count

    # 计算每个类别应该分割出去的数量
    split_counts = {label: int(count * split_ratio) for label, count in total_object_counts.items()}

    # 创建输出目录
    output_file_folder = os.path.join(output_folder)
    os.makedirs(output_file_folder, exist_ok=True)

    # 处理每个XML文件
    for xml_file in os.listdir(input_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(input_folder, xml_file)
            xml_root = parse_xml(xml_path)

            # 创建新的XML文件并复制所需的对象
            output_xml_root = deepcopy(xml_root)
            for obj in xml_root.findall('object'):
                label = obj.find('name').text
                if split_counts[label] > 0:
                    split_counts[label] -= 1
                    xml_root.remove(obj)
                    output_xml_tree = ET.ElementTree(output_xml_root)
                    output_xml_path = os.path.join(output_file_folder, xml_file)
                    output_xml_tree.write(output_xml_path)


def delete_duplicate_files(folder1, folder2):
    # 获取文件夹中的所有文件名
    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = set(os.listdir(folder2))

    # 找到重名的文件
    duplicate_files = files_in_folder1.intersection(files_in_folder2)

    # 删除其中一个文件夹中的重名文件
    for file_name in duplicate_files:
        file_path1 = os.path.join(folder1, file_name)

        if os.path.exists(file_path1):
            os.remove(file_path1)


if __name__ == '__main__':
    # 示例用法
    input_folder = '../train_xmls'
    output_folder = '../test_xmls'
    split_ratio = 0.3
    split_and_save_xmls(input_folder, output_folder, split_ratio)
    delete_duplicate_files(input_folder, output_folder)
