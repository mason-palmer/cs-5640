import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Number of classes (e.g., N classes + 1 for background)
num_classes = 2  # Example: one object class + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoaderb
import os
import numpy as np
import torch
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        boxes = self.parse_annotation(ann_path)
        
        # Convert everything into tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Example: all objects have label 1
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def parse_annotation(self, ann_path):
        # Parse annotation file (e.g., XML, JSON) to extract bounding boxes
        # Example: Return a list of [xmin, ymin, xmax, ymax]
        return [[50, 50, 100, 100], [150, 150, 200, 200]]  # Replace with actual parsing logic

    def __len__(self):
        return len(self.imgs)

# Define transformations (if needed)
dataset = CustomDataset(root="dataset", transforms=None)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


import torch.optim as optim
from torch.utils.data import DataLoader

# Set up optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")


torch.save(model.state_dict(), "faster_rcnn_model.pth")
