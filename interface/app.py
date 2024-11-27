# from google.colab import drive
# drive.mount('/content/drive')
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import Metadata
from detectron2 import model_zoo


import torch
import gradio as gr
import cv2
import numpy as np

# Set up Detectron2 model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) # config file used to train the model
# cfg.MODEL.WEIGHTS = '/content/drive/My Drive/model_final.pth'  # Path to model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for inference
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Set number of classes
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Set device
predictor = DefaultPredictor(cfg)

# Define class names
class_names = ['road traffic', 'bicycles', 'buses', 'crosswalks', 'fire hydrants',
               'motorcycle', 'traffic lights', 'vehicles']
my_metadata = Metadata()
my_metadata.set(thing_classes=class_names)


#=====================================================================
# CREATE IMAGE/VIDEO PROCESSING FUNCTION AND SET UP GRADIO INTERFACE
#=====================================================================



# Function to process video frames
def predict_and_display_frame(frame):
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    outputs = predictor(im)  # Run object detection
    v = Visualizer(im[:, :, ::-1], scale=1.0, instance_mode=0, metadata=my_metadata)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]  # Return processed frame in RGB format

def live_tracking(video_input):
    cap = cv2.VideoCapture(video_input)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = predict_and_display_frame(frame)  # Process frame
        yield processed_frame  # Yield processed frame for Gradio

def detect_objects_in_image(image):
    # Convert the image to the format required by the predictor
    im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return predict_and_display_frame(im)



# Set up Gradio interface for live tracking
with gr.Blocks() as interface:
    gr.Markdown("# Live Object Detection with Detectron2")
    gr.Markdown("### Capture from your webcam, upload a video file, or upload an image")

    with gr.Row():
        video_input = gr.Video(label="Input Video")  # Video input
        start_button = gr.Button("Detect Objects from Uploaded Video")

    with gr.Row():
        image_input = gr.Image(label="Input Image", type="pil")  # Image input
        detect_button = gr.Button("Detect Objects in Image")

    with gr.Row():
        output_image_video = gr.Image(label="Processed Video Output", type="numpy")  # Output for video
        output_image_img = gr.Image(label="Processed Image Output", type="numpy")  # Output for image

    start_button.click(
        live_tracking,
        inputs=video_input,
        outputs=output_image_video
    )

    detect_button.click(
        detect_objects_in_image,
        inputs=image_input,
        outputs=output_image_img
    )

    gr.Markdown("### Instructions:")
    gr.Markdown("1. Choose to capture from your webcam or upload a video file.")
    gr.Markdown("2. Click the 'Start Detection' button to begin object detection on the video.")
    gr.Markdown("3. Upload an image and click 'Detect Objects in Image' to process it.")

# Launch the Gradio app
interface.launch(debug=True)






