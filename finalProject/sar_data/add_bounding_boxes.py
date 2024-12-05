import pandas as pd
import numpy as np

# The train set doesnt have bounding boxes for whatever reason so I will find the average size of the bounding boxes in the validation set and use that to add bounding boxes to the train set.

# Find the average bounding box size in the validation set


validation_set_path = 'validation.csv'
df = pd.read_csv(validation_set_path)

valid_bboxes = df.dropna(subset=['top', 'left', 'bottom', 'right'])

# Calculate bounding box width and height
valid_bboxes['bbox_width'] = valid_bboxes['right'] - valid_bboxes['left']
valid_bboxes['bbox_height'] = valid_bboxes['bottom'] - valid_bboxes['top']

# Calculate average and standard deviation for bounding box width and height
average_width = valid_bboxes['bbox_width'].mean()
std_width = valid_bboxes['bbox_width'].std()

average_height = valid_bboxes['bbox_height'].mean()
std_height = valid_bboxes['bbox_height'].std()


print(f"Average Width: {average_width}, Standard Deviation Width: {std_width}")
print(f"Average Height: {average_height}, Standard Deviation Height: {std_height}")


def update_missing_bounding_boxes(csv_file, output_csv, bbox_width, bbox_height):
    """
    Updates a CSV file by filling missing bounding box fields with estimated sizes.

    Args:
        csv_file (str): Path to the input CSV file.
        output_csv (str): Path to save the updated CSV file.
        bbox_width (int): Estimated bounding box width in pixels.
        bbox_height (int): Estimated bounding box height in pixels.
    """
    # Load the CSV
    df = pd.read_csv(csv_file)

    # Fill missing bounding box values
    def fill_bbox(row):
        if pd.isna(row['top']) or pd.isna(row['left']) or pd.isna(row['bottom']) or pd.isna(row['right']):
            center_row = row['detect_scene_row']
            center_col = row['detect_scene_column']

            top = max(0, center_row - bbox_height // 2)
            left = max(0, center_col - bbox_width // 2)
            bottom = center_row + bbox_height // 2
            right = center_col + bbox_width // 2

            return pd.Series({'top': top, 'left': left, 'bottom': bottom, 'right': right})
        else:
            # Return existing values if they're already present
            return pd.Series({'top': row['top'], 'left': row['left'], 'bottom': row['bottom'], 'right': row['right']})

    # Apply function to rows with missing values
    updated_bboxes = df.apply(fill_bbox, axis=1)
    df['top'] = updated_bboxes['top']
    df['left'] = updated_bboxes['left']
    df['bottom'] = updated_bboxes['bottom']
    df['right'] = updated_bboxes['right']

    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

# Update the CSV file
update_missing_bounding_boxes('train.csv', 'new_train.csv', average_width, average_height)

