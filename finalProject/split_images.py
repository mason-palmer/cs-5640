import cv2
import os
import numpy as np
import glob

# Function to split images and their bounding boxes
def split_images_and_labels(image_path, label_path, output_dir, grid_size=(2, 2)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} not found.")

    # Get image dimensions
    height, width = image.shape[:2]

    # Load YOLO labels
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # Parse YOLO labels
    bounding_boxes = []
    for label in labels:
        data = label.strip().split()
        cls, x_center, y_center, bbox_width, bbox_height = map(float, data)
        bounding_boxes.append([cls, x_center, y_center, bbox_width, bbox_height])

    # Split the image into smaller patches
    rows, cols = grid_size
    patch_height, patch_width = height // rows, width // cols

    patch_counter = 0
    for i in range(rows):
        for j in range(cols):
            # Compute patch boundaries
            start_y, end_y = i * patch_height, (i + 1) * patch_height
            start_x, end_x = j * patch_width, (j + 1) * patch_width

            # Extract image patch
            patch = image[start_y:end_y, start_x:end_x]

            # Save image patch
            patch_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_{patch_counter}.png"
            patch_path = os.path.join(output_dir, patch_filename)
            cv2.imwrite(patch_path, patch)

            # Adjust and save the corresponding labels for the patch
            patch_labels = []
            for box in bounding_boxes:
                cls, x_center, y_center, bbox_width, bbox_height = box

                # Convert to pixel coordinates
                x_center_abs = x_center * width
                y_center_abs = y_center * height
                bbox_width_abs = bbox_width * width
                bbox_height_abs = bbox_height * height

                # Check if bounding box is within the current patch
                if (start_x <= x_center_abs <= end_x) and (start_y <= y_center_abs <= end_y):
                    # Adjust bounding box for the new patch
                    new_x_center = (x_center_abs - start_x) / patch_width
                    new_y_center = (y_center_abs - start_y) / patch_height
                    new_bbox_width = bbox_width_abs / patch_width
                    new_bbox_height = bbox_height_abs / patch_height

                    patch_labels.append(f"{cls} {new_x_center} {new_y_center} {new_bbox_width} {new_bbox_height}")

            # Save labels for this patch
            patch_label_filename = f"{os.path.splitext(os.path.basename(label_path))[0]}_{patch_counter}.txt"
            patch_label_path = os.path.join(output_dir, patch_label_filename)
            with open(patch_label_path, 'w') as f:
                for label in patch_labels:
                    f.write(label + '\n')

            patch_counter += 1

# Function to combine smaller images and labels into the original size
def combine_labels(txt_files, output_txt_file, original_size, grid_size=(2, 2)):
    rows, cols = grid_size
    original_height, original_width = original_size
    patch_height, patch_width = original_height // rows, original_width // cols

    txts = [x.split("/")[-1] for x in txt_files]

    tl_x = original_width
    tl_y = original_height
    br_x = 0
    br_y = 0

    labels = []
    patch_counter = 0
    for i in range(rows):
        for j in range(cols):
            if(output_txt_file.split("/")[-1].replace('.txt',f"_{patch_counter}.txt") in txts): 
                with open(txt_files[txts.index(output_txt_file.split("/")[-1].replace('.txt',f"_{patch_counter}.txt"))], 'r') as f:
                    patch_labels = f.readlines()
                    for patch_label in patch_labels:
                        data = patch_label.strip().split()
                        cls, x_center, y_center, bbox_width, bbox_height = map(float, data)

                        absolute_x = (x_center * patch_width) + (j * patch_width)
                        absolute_y = (y_center * patch_height) + (i * patch_height)
                        absolute_width = bbox_width * patch_width
                        absolute_height = bbox_height * patch_height

                        original_x = absolute_x/original_width
                        original_y = absolute_y/original_height
                        bbox_original_width = absolute_width/original_width
                        bbox_original_height = absolute_height/original_height

                        labels.append(f"{cls} {original_x} {original_y} {bbox_original_width} {bbox_original_height}")
            patch_counter += 1

    if(br_x != 0):
        bee_box_width = br_x - tl_x
        bee_box_height = br_y - tl_y
        original_x = (tl_x + bee_box_width/2)/original_width
        original_y = (tl_y + bee_box_height/2)/original_height
        bbox_original_width = bee_box_width/original_width
        bbox_original_height = bee_box_height/original_height

        labels.append(f"6 {original_x} {original_y} {bbox_original_width} {bbox_original_height}")

    with open(os.path.join(os.getcwd(), output_txt_file), 'w+') as f:
        for label in labels:
            f.write(label + '\n')
