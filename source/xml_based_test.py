import os
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from train_v2 import EnsembleModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to your folder containing the XML files
folder_path = '../test_xmls_split'

# Define the number of classes for the output layer of your network
num_classes = 5

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        coords = {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
    path = '../official_test/{}'.format(root.find('path').text)
    name = root.find('path').text
    return path, coords, name, int(bndbox.find('xmin').text), int(bndbox.find('ymin').text), int(
        bndbox.find('xmax').text), int(bndbox.find('ymax').text)


# Initialize the model
model = EnsembleModel(num_classes)
# Load the model weights (use the appropriate file path)
model.load_state_dict(torch.load('../models/model_for_test.pth'))
model = model.to(device)
model.eval()

# Define a map from your predicted indices to class names
label_map = {0: 'good', 1: 'broke', 2: 'lose', 3: 'uncovered', 4: 'circle'}

# Main loop to process each XML file in the folder
for xml_file in os.listdir(folder_path):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(folder_path, xml_file)

        # Parse the XML file
        image_path, bndbox_coords, image_name, xmin, ymin, xmax, ymax = parse_xml(xml_path)

        # Load the image and preprocess it
        image = Image.open(image_path).convert('RGB')
        object_image = image.crop((bndbox_coords['xmin'], bndbox_coords['ymin'],
                                   bndbox_coords['xmax'], bndbox_coords['ymax']))
        test_image = transform(object_image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            test_tensor = test_image.to(device)
            mobilenet_out, resnet_out = model(test_tensor)
            combined_out = mobilenet_out + resnet_out
            probabilities = nn.functional.softmax(combined_out, dim=1)
            predicted_index = probabilities.argmax(1).item()
            confidence = probabilities.max(1).values.item()
            predicted_label = {0: 'good', 1: 'broke', 2: 'lose', 3: 'uncovered', 4: 'circle'}
            print('{}  {}  {}  {}  {}  {}  {}'.format(image_name,
                                                      predicted_label[predicted_index],
                                                      confidence,
                                                      xmin, ymin, xmax, ymax
                                                      ))
