import torch
import gradio as gr
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from detectron2 import model_zoo

# Set up Detectron2 model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) # config file used to train the model
cfg.MODEL.WEIGHTS = '/content/drive/My Drive/model_final.pth'  # Path to model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for inference
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Set number of classes
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set device
predictor = DefaultPredictor(cfg)

# Define class names
class_names = ['road traffic', 'bicycles', 'buses', 'crosswalks', 'fire hydrants',
               'motorcycle', 'traffic lights', 'vehicles']
my_metadata = Metadata()
my_metadata.set(thing_classes=class_names)