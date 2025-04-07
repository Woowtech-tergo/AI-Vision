import gradio as gr
import os
import time
import cv2
from ultralytics import YOLO

def process_video(video_input):
    """
    Função para processar o vídeo com YOLOv8n.
    Recebe o caminho temporário do vídeo via 'video_input'.
    Retorna:
      - o caminho do vídeo resultante (para exibir em gr.Video),
      - uma mensagem de status,
      - o mesmo caminho (para permitir download em gr.File).
    """
    if video_input is None:
        return None, "Nenhum arquivo de vídeo enviado", None

    start_time = time.time()

    # Carrega o modelo YOLOv8n
    model = YOLO("yolov8n.pt")

    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        return None, "Não foi possível abrir o vídeo", None

    # Prepara o vídeo de saída
    out_path = "processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Loop para ler cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Roda inferência do YOLO
        results = model(frame)
        # Desenha as detecções no frame
        annotated_frame = results[0].plot()
        # Salva no vídeo de saída
        out.write(annotated_frame)

    cap.release()
    out.release()
    end_time = time.time()
    elapsed = end_time - start_time
    msg = f"Processamento concluído em {elapsed:.2f} segundos."

    # Retorna (vídeo_para_exibir, mensagem, vídeo_para_download)
    return out_path, msg, out_path

# Constrói a interface minimalista do Gradio
with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Video Detector (Minimal)")

    # Entrada: um componente de vídeo e um botão
    with gr.Row():
        input_video = gr.Video(label="Carregue o vídeo aqui", type="mp4")
        process_button = gr.Button("Processar")

    # Saída: vídeo processado, mensagem, e botão de download
    output_video = gr.Video(label="Resultado")
    status = gr.Textbox(label="Status")
    download_file = gr.File(label="Download do vídeo", file_count="single")

    # Evento: quando clica em "Processar", chama 'process_video'
    process_button.click(
        fn=process_video,
        inputs=[input_video],
        outputs=[output_video, status, download_file]
    )

if __name__ == "__main__":
    # Inicia o Gradio localmente na porta 7860
    # Se estiver no Railway, a porta pode ser substituída pela variável de ambiente PORT.
    demo.launch(server_name="0.0.0.0", server_port=7860)
