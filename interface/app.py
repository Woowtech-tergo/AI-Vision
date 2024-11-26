import gradio as gr

def video_player(video):
    return video

app = gr.Interface(
    fn=video_player,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Video(label="Video Player"),
    title="Leitor de Vídeo",
    description="Carregue um vídeo para reproduzi-lo."
)

if __name__ == "__main__":
    app.launch()