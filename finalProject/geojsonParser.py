import json
import os
import cv2

#assuming geojson is in same folder, modify if needed
with open('xView_train.geojson', 'r') as file:
    data = json.load(file)

#should be the path to the train_images, can be relative path
folder_path = "./train_images"

for feature in data['features']:

    #get relevant info (making sure numbers involved in calculations are int's)
    image_name = feature['properties']['image_id']
    min_x, min_y, max_x, max_y = [int(x) for x in (feature['properties']['bounds_imcoords']).split(',')]
    classification = feature['properties']['type_id']

    #load linked image to get dimensions
    image = cv2.imread(os.path.join(folder_path, image_name))
    if image is None:
        raise ValueError(f"image at {os.path.join(folder_path, image_name)} not found")

    img_height, img_width = image.shape[:2]

    #get the relative coordinates for bounding box in center point width and height
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    bbox_x = (min_x + bbox_width/2)/img_width
    bbox_y = (min_y + bbox_height/2)/img_height
    bbox_width = bbox_width/img_width
    bbox_height = bbox_height/img_height

    #add bbox to yolo label txt
    with open(os.path.join(folder_path, image_name.replace('.tif', '.txt')), 'a+') as label_file:
        label_file.write(f"{classification} {bbox_x} {bbox_y} {bbox_width} {bbox_height}\n")