import os
import json
import pandas as pd
import rasterio

def convert_csv_to_coco_format(csv_file, image_dir, output_json):
    """
    Converts a CSV file to COCO format JSON.

    Args:
        csv_file (str): Path to the input CSV file.
        image_dir (str): Directory containing the images.
        output_json (str): Path to save the COCO format JSON file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize COCO JSON structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "vessel", "supercategory": "object"},
            {"id": 2, "name": "fishing_vessel", "supercategory": "object"}
        ]
    }

    annotation_id = 1  # Unique ID for each annotation
    image_id_map = {}  # Map to track image IDs
    image_id_counter = 1  # Unique ID for each image

    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        scene_id = row['scene_id']
        # image_path = os.path.join(image_dir, f"{scene_id}_stacked.tif")
        image_path = f"{image_dir}/{scene_id}_stacked.tif"

        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image for scene_id '{image_path}' not found. Skipping.")
            continue

        # Get image dimensions using rasterio
        try:
            with rasterio.open(image_path) as src:
                width = src.width
                height = src.height
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            continue

        # Add image metadata to COCO format (if not already added)
        if scene_id not in image_id_map:
            coco_format["images"].append({
                "id": image_id_counter,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height
            })
            image_id_map[scene_id] = image_id_counter
            image_id_counter += 1

        # Determine the category ID based on is_vessel and is_fishing
        is_vessel = row['is_vessel']
        is_fishing = row['is_fishing']

        if not is_vessel:
            continue  # Skip non-vessel annotations
        elif not is_fishing:
            category_id = 1  # Vessel
        else:
            category_id = 2  # Fishing vessel

        # Extract bounding box coordinates
        top = row['top']
        left = row['left']
        bottom = row['bottom']
        right = row['right']

        # Ensure bounding box values are valid
        if pd.isna(top) or pd.isna(left) or pd.isna(bottom) or pd.isna(right):
            print(f"Missing bounding box values for scene_id '{scene_id}'. Skipping annotation.")
            continue

        # Convert to COCO format (x, y, width, height)
        bbox_width = right - left
        bbox_height = bottom - top
        coco_bbox = [left, top, bbox_width, bbox_height]

        # Add annotation to COCO format
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id_map[scene_id],
            "category_id": category_id,
            "bbox": coco_bbox,
            "area": bbox_width * bbox_height,  # Area of the bounding box
            "iscrowd": 0
        })
        annotation_id += 1

    # Save COCO format JSON
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO format JSON saved to {output_json}")

if __name__ == "__main__":
    convert_csv_to_coco_format(
        csv_file='validation.csv',
        image_dir='./ValData',
        output_json='validation_coco.json'
    )
