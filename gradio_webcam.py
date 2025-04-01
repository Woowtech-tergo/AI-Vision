import gradio as gr
from gradio_webrtc import WebRTC


def detection(image, conf_threshold=0.3):
    ...


with gr.Blocks() as demo:
    image = WebRTC(label="Stream", mode="send-receive", modality="video")
    conf_threshold = gr.Slider(
        label="Confidence Threshold",
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        value=0.30,
    )
    image.stream(
        fn=detection,
        inputs=[image, conf_threshold],
        outputs=[image], time_limit=10
    )

if __name__ == "__main__":
    demo.launch()
