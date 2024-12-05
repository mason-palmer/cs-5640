import os
import pandas as pd
import rasterio

def convert_csv_to_yolo_labels(csv_file, output_dir, image_dir):
    """
    Reads the CSV file and writes YOLO label files for each scene_id.

    Args:
        csv_file (str): Path to the input CSV file.
        output_dir (str): Directory where YOLO label files will be saved.
        image_dir (str): Directory containing images for which metadata will be extracted.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group by scene_id
    grouped = df.groupby('scene_id')

    for scene_id, group in grouped:
        # Construct the image path
        image_path = os.path.join(image_dir, f"{scene_id}_stacked.tif")

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image for scene_id '{scene_id}' not found. Skipping.")
            continue

        print(f"Processing {image_path}...")

        # Extract image dimensions using rasterio
        try:
            with rasterio.open(image_path) as src:
                img_width = src.width
                img_height = src.height
        except Exception as e:
            print(f"Failed to open image for scene_id '{scene_id}': {e}. Skipping.")
            continue

        label_lines = []
        for _, row in group.iterrows():
            # Assign class_id based on is_vessel and is_fishing
            is_vessel = row['is_vessel']
            is_fishing = row['is_fishing']

            # Skip non-vessel entries
            if not is_vessel:
                continue
            elif not is_fishing:
                class_id = 1
            else:
                class_id = 2

            # Extract and normalize bounding box coordinates
            top = row['top']
            left = row['left']
            bottom = row['bottom']
            right = row['right']

            x_center = (left + right) / 2.0 / img_width
            y_center = (top + bottom) / 2.0 / img_height
            bbox_width = (right - left) / img_width
            bbox_height = (bottom - top) / img_height

            # Ensure coordinates are within [0,1]
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            bbox_width = min(max(bbox_width, 0), 1)
            bbox_height = min(max(bbox_height, 0), 1)

            # Format: <class_id> <x_center> <y_center> <width> <height>
            label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
            label_lines.append(label_line)

        # Write to a .txt file named after the scene_id
        label_file = os.path.join(output_dir, f"{scene_id}.txt")
        with open(label_file, 'w') as f:
            f.write('\n'.join(label_lines))

        print(f"Label file for scene_id '{scene_id}' written to '{label_file}'.")

# Example usage
if __name__ == "__main__":
    csv_file_path = "labels.csv"  # Replace with your CSV file path
    output_directory = "labels"   # Directory to save YOLO label files
    image_directory = "images"    # Directory containing your .tif files

    convert_csv_to_yolo_labels(csv_file_path, output_directory, image_directory)
