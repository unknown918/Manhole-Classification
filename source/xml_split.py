import os
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET


def split_xml_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的每个 XML 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.xml'):
            input_xml_file = os.path.join(input_folder, filename)
            split_xml(input_xml_file, output_folder)


def split_xml(input_xml_file, output_folder):
    # 解析原始 XML 文件
    tree = ET.parse(input_xml_file)
    root = tree.getroot()

    # 获取对象数目
    num_objects = len(root.findall('object'))

    # 遍历每个对象
    for obj_index, obj in enumerate(root.findall('object'), start=1):
        # 创建新的 XML 根元素
        new_root = ET.Element("annotation")

        # 复制原始 XML 中的基本信息（例如文件夹、文件名、路径等）
        folder = ET.SubElement(new_root, "folder")
        folder.text = root.find("folder").text

        filename = ET.SubElement(new_root, "filename")
        filename.text = root.find("filename").text

        path = ET.SubElement(new_root, "path")
        path.text = root.find("path").text

        source = ET.SubElement(new_root, "source")
        database = ET.SubElement(source, "database")
        database.text = root.find("source/database").text

        # 复制图像尺寸信息
        size = ET.SubElement(new_root, "size")
        size_width = ET.SubElement(size, "width")
        size_width.text = root.find("size/width").text
        size_height = ET.SubElement(size, "height")
        size_height.text = root.find("size/height").text
        size_depth = ET.SubElement(size, "depth")
        size_depth.text = root.find("size/depth").text

        # 复制分割信息
        segmented = ET.SubElement(new_root, "segmented")
        segmented.text = root.find("segmented").text

        # 复制当前对象到新的 XML 文件中
        new_root.append(obj)

        # 创建新的 XML 文件并保存，使用对象数目作为前缀
        output_xml_file = os.path.join(output_folder,
                                       f"{num_objects}_{obj.find('name').text}_{filename.text.split('.')[0]}_{obj_index}.xml")
        new_tree = ET.ElementTree(new_root)
        new_tree.write(output_xml_file)

        xml_str = minidom.parseString(ET.tostring(new_root)).toprettyxml(indent="  ")
        with open(output_xml_file, "w") as xml_file:
            xml_file.write(xml_str)


if __name__ == "__main__":
    # 使用示例
    input_folder = "../test_xmls"
    output_folder = "../test"
    split_xml_folder(input_folder, output_folder)
