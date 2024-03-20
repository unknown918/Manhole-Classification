import os
import cv2
import numpy as np
from tqdm import tqdm


def apply_operations(image, operations):
    result_images = []
    for operation in operations:
        if operation == 'origin':
            result_images.append(image.copy())
        elif operation == 'gaussian_blur':
            result_images.append(cv2.GaussianBlur(image, (5, 5), 0))
        elif operation == 'brightness_adjustment':
            result_images.append(cv2.convertScaleAbs(image, alpha=1.2, beta=10))
        elif operation == 'noise_addition':
            noise = np.random.normal(loc=0, scale=25, size=image.shape)
            result_images.append(np.clip(image + noise, 0, 255).astype(np.uint8))
        elif operation == 'rotation':
            result_images.append(np.rot90(image))
        elif operation == 'rotation_and_brightness':
            rotated_image = np.rot90(image)
            result_images.append(cv2.convertScaleAbs(rotated_image, alpha=1.2, beta=10))
    return result_images


def expand_dataset(input_folder, output_folder):
    operations = ['gaussian_blur', 'brightness_adjustment', 'noise_addition', 'rotation', 'rotation_and_brightness']
    for subdir in os.listdir(input_folder):
        input_subdir = os.path.join(input_folder, subdir)
        output_subdir = os.path.join(output_folder, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        for file in tqdm(os.listdir(input_subdir), desc=subdir):
            input_path = os.path.join(input_subdir, file)
            image = cv2.imread(input_path)

            for operation in operations:
                output_images = apply_operations(image, [operation])

                for i, output_image in enumerate(output_images):
                    output_path = os.path.join(output_subdir, f"{operation}_{i}_{file}")
                    cv2.imwrite(output_path, output_image)


if __name__ == "__main__":
    input_folder = "../datasets/train"
    output_folder = "../datasets/train"
    expand_dataset(input_folder, output_folder)

