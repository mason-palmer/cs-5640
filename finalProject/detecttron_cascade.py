from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import cv2
import os

from detectron2.data.datasets import register_coco_instances
import rasterio
import numpy as np
from detectron2.data import detection_utils as utils

# 
# def custom_read_image(file_name, format=None):
#     with rasterio.open(file_name) as src:
#         image = src.read()  # Reads the file as a multi-dimensional array (bands, height, width)

#     # Convert (bands, height, width) to (height, width, bands) for compatibility
#     image = np.transpose(image, (1, 2, 0))

#     if format == "BGR":
#         # Convert to BGR format if required
#         image = image[:, :, ::-1]

#     return image

# combine VV and VH into a single grayscale channel
def custom_read_image(file_name, format=None):
    import rasterio
    import numpy as np

    with rasterio.open(file_name) as src:
        vv = src.read(1)  # VV polarization
        vh = src.read(2)  # VH polarization

        # Combine VV and VH into a single grayscale channel
        grayscale = (vv + vh) / 2

        # Expand to a 3-channel format by duplicating the grayscale channel
        image = np.stack([vv, vh, grayscale], axis=-1)  # (H, W, 3)

        return image

utils.read_image = custom_read_image
config_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# Register the dataset
register_coco_instances("cascade_train_dataset", {}, "train_coco.json", "TrainData")
register_coco_instances("cascade_val_dataset", {}, "validation_coco.json", "ValData")
# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file(config_path))  # Use pre-configured Cascade R-CNN
cfg.MODEL.PIXEL_MEAN = [123.0, 123.0, 123.0]  # Two channels (VV and VH)
cfg.MODEL.PIXEL_STD = [58.0, 58.0, 58.0]     # Two channels
cfg.DATASETS.TRAIN = ("cascade_train_dataset",)
cfg.DATASETS.TEST = ("cascade_val_dataset",)  # For evaluation
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)  # Pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # Adjust based on your dataset
cfg.SOLVER.MAX_ITER = 3000  # Number of training iterations
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Change this to the number of classes in your dataset (excluding background)
cfg.MODEL.DEVICE = "cuda"

def main():
    # Output directory for saving checkpoints and logs
    cfg.OUTPUT_DIR = "./output_cascade_rcnn"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train the model
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Create evaluator and test loader
    evaluator = COCOEvaluator("cascade_val_dataset", cfg, False, output_dir="./output_cascade_rcnn")
    val_loader = build_detection_test_loader(cfg, "cascade_val_dataset")

    # Perform evaluation
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # Load the trained model
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = f"{cfg.OUTPUT_DIR}/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for predictions
    predictor = DefaultPredictor(cfg)

    # Load and predict on a new image
    image_path = "test.tif"
    image = cv2.imread(image_path)
    outputs = predictor(image)

    # Visualize predictions
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("cascade_test_dataset"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Result", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    main()