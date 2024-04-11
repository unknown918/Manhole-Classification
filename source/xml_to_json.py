import json
import os
import xml.etree.ElementTree as ET


def xml_to_json(xml_folder):
    json_data = {}
    for filename in os.listdir(xml_folder):
        if filename.endswith('.xml'):
            xml_file = os.path.join(xml_folder, filename)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            image_name = root.find('filename').text
            all_points_x = []
            all_points_y = []
            for obj in root.findall('object'):
                try:
                    for pt in obj.find('bndbox'):
                        if pt.tag in ('xmin', 'xmax'):
                            all_points_x.append(int(pt.text))
                        elif pt.tag in ('ymin', 'ymax'):
                            all_points_y.append(int(pt.text))
                except AttributeError:
                    # 如果没有找到<bndbox>元素，则跳过当前对象
                    continue

            image_width = max(all_points_x) - min(all_points_x)
            image_height = max(all_points_y) - min(all_points_y)
            image_size = image_width * image_height

            regions = {
                '0': {
                    'shape_attributes': {
                        'name': 'polygon',
                        'all_points_x': all_points_x,
                        'all_points_y': all_points_y
                    },
                    'region_attributes': {}
                }
            }

            json_data[image_name] = {
                'fileref': "",
                'size': image_size,
                'filename': image_name,
                'base64_img_data': "",
                'file_attributes': {},
                'regions': regions
            }

    return json_data


# 指定包含XML文件的文件夹路径
xml_folder = "../train_xmls_split"

# 将XML转换为JSON
json_data = xml_to_json(xml_folder)

# 将JSON写入文件
with open("../train.json", "w") as json_file:
    json.dump(json_data, json_file, indent=4)
