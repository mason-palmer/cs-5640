import os
from stack_polarizations import combine_sar_channels 
from create_yolo_labels import convert_csv_to_yolo_labels

raw_dir = 'D:/ML_DataSets/SAR/Validation'
processed_dir = 'C:/Users/Brad/Documents/Masters/ANNs/cs-5640/trainData'
labels_csv = 'C:/Users/Brad/Documents/Masters/ANNs/cs-5640/validation.csv'

def process_scene(scene_dir, processed_dir):
    # Add stacked image to processed_dir
    scene_id = os.path.basename(scene_dir)
    combine_sar_channels(f"{scene_dir}/VV_dB.tif", f"{scene_dir}/VH_dB.tif", f"{processed_dir}/{scene_id}_stacked.tif")

    # Add label file to processed_dir
    # convert_csv_to_yolo_labels(f"{scene_dir}/labels.csv", processed_dir, scene_dir)

    # TODO: Split images



if __name__ == "__main__":
    # Stack Images
    # for scene in os.listdir(raw_dir):
    #     # scene_dir = os.path.join(raw_dir, scene)
    #     scene_dir = f"{raw_dir}/{scene}"
    #     if os.path.isdir(scene_dir):
    #         print(scene_dir)
    #         if os.path.isdir(scene_dir):
    #             # Process the scene
    #             process_scene(scene_dir, processed_dir)
    
    # create data labels
    convert_csv_to_yolo_labels(labels_csv, processed_dir, processed_dir)