import os
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

# Interface do Gradio
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

if __name__ == "__main__":
    # Use a porta fornecida pela variável de ambiente PORT
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
