from ultralytics import YOLO
from split_images import *
import torch
from show import *
import cv2
import os
import glob
import shutil

run = "run1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' if not torch.backends.mps.is_built() else "mps")

model = YOLO("output/test3/weights/best.pt")

all_tifs = glob.glob(os.path.join("./train_images", "*.tif")) 
            # + glob.glob(os.path.join("./val_images", "*.tif"))

for tif in all_tifs:
    split_images_and_labels(tif, tif.replace("tif","txt"), f"./runs/{run}", (4, 6))
    img_tifs = glob.glob(os.path.join(f"./runs/{run}/{tif.split('/')[-1].replace('.tif','')}*.tif"))
    
    for image in img_tifs:
        results = model(image, save=True, save_txt=True, device=device, project='runs/', name=run, exist_ok=True)

    txt_files = glob.glob(os.path.join(f"./runs/{run}/labels/{png.split('/')[-1].replace('.tif','')}*.txt"))
    os.makedirs(os.path.join(os.getcwd(), f"runs/{run}/results"), exist_ok=True)
    combine_labels(txt_files, f"runs/{run}/results/{tif.split('/')[-1].replace('.tif','.txt')}", cv2.imread(tif).shape[:2], (4,6))
    shutil.copyfile(tif, f"./runs/{run}/results/{tif.split('/')[-1]}")
    draw_bboxes(f"./runs/{run}/results/{tif.split('/')[-1]}", f"runs/{run}/results/{tif.split('/')[-1].replace('.tif','.txt')}")