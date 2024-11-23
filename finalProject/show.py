import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to denormalize the bounding box coordinates
def denormalize_bbox(bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = bbox
    x_center = int(x_center * img_width)
    y_center = int(y_center * img_height)
    width = int(width * img_width)
    height = int(height * img_height)
    
    # Get the top-left corner of the bounding box
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    
    # Get the bottom-right corner of the bounding box
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return class_id, x1, y1, x2, y2

# Function to draw bounding boxes on the image
def draw_bboxes(img_path, label_path):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    
    # Get image dimensions
    img_height, img_width, _ = img.shape
    
    # Load labels (bounding boxes)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    # Process each bounding box
    for label in labels:
        bbox = list(map(float, label.strip().split()))
        class_id, x1, y1, x2, y2 = denormalize_bbox(bbox, img_width, img_height)

        # Draw bounding box (green rectangle)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        
        # Optionally, draw the class id on the box
        cv2.putText(img, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the image with bounding boxes

    dpi = mpl.rcParams['figure.dpi']
    im_data = img
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f"{img_path.replace('.png','')}_bboxes.png", bbox_inches='tight')
    # plt.show()