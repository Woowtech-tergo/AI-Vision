from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
import time
import os

import deep_sort.deep_sort.deep_sort as ds

import gradio as gr

# Import YOLO DeepSort functions
from Modelos.YOLOv8DeepSortTracking.yolo_deepsort import (
    get_detectable_classes as yolo_deepsort_get_detectable_classes,
    stop_processing as yolo_deepsort_stop_processing,
    start_processing as yolo_deepsort_start_processing,
    putTextWithBackground,
    extract_detections,
    detect_and_track,
)


class App:
    def __init__(self):
        # Lista de modelos disponíveis
        self.model_list = ["YOLOv8DeepSort"]  # Caso adicione outros modelos, inclua-os aqui

        # Mapeamento de funções por modelo
        self.model_functions = {
            "YOLOv8DeepSort": {
                "get_detectable_classes": yolo_deepsort_get_detectable_classes,
                "start_processing": yolo_deepsort_start_processing,
                "stop_processing": yolo_deepsort_stop_processing,
                "model_file": "yolov8n.pt",  # Modelo padrão para YOLOv8DeepSort
            },
            # Adicione outros modelos aqui
        }

        # Inicializa a interface Gradio
        with gr.Blocks() as self.demo:
            # Título
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos
                Baseado em OpenCV + YOLOv8 + DeepSort
                """
            )

            # Carregar vídeo
            with gr.Row():
                self.video_input = gr.Video(label="Vídeo de Entrada")  # Controle para upload de vídeo

            # Selecionar modelo
            with gr.Row():
                self.model_dropdown = gr.Dropdown(
                    choices=self.model_list, label="Modelo", value=self.model_list[0]
                )

            # Estados para armazenar o vídeo de entrada e o caminho do vídeo processado
            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            # Botão para carregar o vídeo
            with gr.Row():
                self.load_video_button = gr.Button("Carregar Vídeo")

            # Após carregar o vídeo, mostram-se as opções adicionais
            with gr.Column(visible=False) as self.options_column:
                with gr.Row():
                    self.detect_class_dropdown = gr.Dropdown(
                        choices=[], label="Classe"
                    )
                with gr.Row():
                    self.start_button = gr.Button("Iniciar Processamento")
                    self.stop_button = gr.Button("Interromper Processamento")
                with gr.Row():
                    self.processing_message = gr.Textbox(
                        label="Mensagem",
                        visible=False
                    )
                with gr.Row():
                    self.output_video = gr.Video(label="Vídeo Processado")
                with gr.Row():
                    self.download_button = gr.Button("Download", visible=False)
                    self.processing_time_label = gr.Textbox(
                        label="Tempo de Processamento"
                    )
                    self.file_info = gr.Markdown(
                        value="", visible=False
                    )

            # Eventos
            self.load_video_button.click(
                fn=self.load_video,
                inputs=[self.video_input, self.model_dropdown],
                outputs=[
                    self.input_video_state,
                    self.options_column,
                    self.detect_class_dropdown,
                ],
            )

            self.model_dropdown.change(
                fn=self.update_detect_classes,
                inputs=[self.model_dropdown],
                outputs=[self.detect_class_dropdown],
            )

            self.start_button.click(
                fn=self.on_start_processing,
                inputs=[
                    self.input_video_state,
                    self.detect_class_dropdown,
                    self.model_dropdown,
                ],
                outputs=[
                    self.output_video,
                    self.processing_time_label,
                    self.output_video_path_state,
                    self.processing_message,
                    self.download_button,
                    self.file_info,
                ],
            )

            self.download_button.click(
                fn=self.on_download_button_clicked,
                inputs=[self.output_video_path_state],
                outputs=gr.File(file_count="single"),
            )

            self.stop_button.click(
                fn=self.on_stop_processing,
                inputs=[],
                outputs=[],
            )

    def load_video(self, video_input, model_name):
        # Armazena o vídeo no estado e torna as opções visíveis
        detect_classes = self.get_detect_classes(model_name)
        return (
            video_input,
            gr.update(visible=True),
            gr.update(choices=detect_classes, value=detect_classes[0]),
        )

    def update_detect_classes(self, model_name):
        # Atualiza a lista de classes detectáveis ao mudar o modelo
        detect_classes = self.get_detect_classes(model_name)
        return gr.update(choices=detect_classes, value=detect_classes[0])

    def get_detect_classes(self, model_name):
        # Obtém as classes detectáveis do modelo selecionado
        model_info = self.model_functions.get(model_name)
        if model_info:
            get_classes_func = model_info["get_detectable_classes"]
            model_file = model_info["model_file"]
            detect_classes = get_classes_func(model_file)
            return detect_classes
        else:
            return []

    def on_start_processing(self, input_video_path, detect_class, model_name):
        # Processa o vídeo
        model_info = self.model_functions.get(model_name)
        if model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info["model_file"]

            start_time = time.time()
            output_dir = tempfile.mkdtemp()
            output_video_path, _ = start_processing_func(
                input_video_path, output_dir, detect_class, model_file
            )
            end_time = time.time()

            # Converte o Path para string
            output_video_path_str = str(output_video_path)

            # Calcula o tempo de processamento
            processing_time = end_time - start_time
            hours, rem = divmod(processing_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            # Obter o tamanho do arquivo
            file_size = os.path.getsize(output_video_path_str)
            file_size_mb = file_size / (1024 * 1024)
            file_size_str = f"{file_size_mb:.2f} MB"

            # Atualiza a mensagem e torna o botão de download visível
            processing_message = "Seu vídeo está pronto para download."
            file_info = f"**Tamanho do arquivo:** {file_size_str}"

            return (
                output_video_path_str,
                time_str,
                output_video_path_str,
                gr.update(value=processing_message, visible=True),
                gr.update(visible=True),
                gr.update(value=file_info, visible=True),
            )
        else:
            return None, "Modelo não encontrado", None, None, None, None

    def on_download_button_clicked(self, output_video_path):
        # Retorna o caminho do vídeo processado para download
        return str(output_video_path)

    def on_stop_processing(self):
        # Interrompe o processamento
        model_name = self.model_dropdown.value
        model_info = self.model_functions.get(model_name)
        if model_info:
            stop_processing_func = model_info["stop_processing"]
            stop_processing_func()


if __name__ == "__main__":
    app = App()
    app.demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
