import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('')

# Directory of the folder containing images
directory = ''

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # assuming images are .jpg or .png
        # Construct the full path to the image
        img_path = os.path.join(directory, filename)
        # Load the image
        img = image.load_img(img_path,
                             target_size=(224, 224))  # adjust target_size to match the model's expected input size
        # Preprocess the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        img_array /= 255.0  # Scale the image (if your model requires scaling)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]  # Assuming your model outputs one-hot encoded predictions

        print(f'File: {filename}, Predicted class: {predicted_class}')
    else:
        continue  # if not an image file, skip to the next file
