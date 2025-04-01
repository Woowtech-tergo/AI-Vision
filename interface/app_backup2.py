# app.py

from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
import time
import os

import deep_sort.deep_sort.deep_sort as ds

import gradio as gr
from gradio_webrtc import WebRTC

# Importar o módulo yolo_deepsort completo
from Modelos.YOLOv8DeepSortTracking import yolo_deepsort

class App:
    def __init__(self):
        # Lista de modelos disponíveis
        self.model_list = ["YOLOv8DeepSort"]  # Caso adicione outros modelos, inclua-os aqui

        # Mapeamento de funções por modelo
        self.model_functions = {
            "YOLOv8DeepSort": {
                "get_detectable_classes": yolo_deepsort.get_detectable_classes,
                "start_processing": yolo_deepsort.start_processing,
                "stop_processing": yolo_deepsort.stop_processing,
                "process_webcam_frame": yolo_deepsort.process_webcam_frame,
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

            # Selecionar fonte de entrada
            with gr.Row():
                self.input_source_radio = gr.Radio(
                    choices=["Arquivo de Vídeo", "Webcam"],
                    label="Fonte de Entrada",
                    value="Arquivo de Vídeo"
                )

            # Carregar vídeo ou usar webcam
            with gr.Row():
                self.video_input = gr.Video(label="Vídeo de Entrada", visible=True)
                # self.webcam_input = WebRTC(
                #     label="Webcam",
                #     mode="send-receive",
                #     rtc_configuration={},
                #     visible=False,
                # )
                self.webcam_input = WebRTC(label="Webcam",
                                           visible=False)

            # Selecionar modelo
            with gr.Row():
                self.model_dropdown = gr.Dropdown(
                    choices=self.model_list, label="Modelo", value=self.model_list[0]
                )

            # Estados para armazenar o vídeo de entrada e o caminho do vídeo processado
            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            # Botão para carregar o vídeo ou iniciar a webcam
            with gr.Row():
                self.load_video_button = gr.Button("Carregar Vídeo / Iniciar Webcam")

            # Após carregar o vídeo ou iniciar a webcam, mostram-se as opções adicionais
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
            self.input_source_radio.change(
                fn=self.update_input_source_visibility,
                inputs=[self.input_source_radio],
                outputs=[self.video_input, self.webcam_input],
            )

            self.load_video_button.click(
                fn=self.load_video_or_webcam,
                inputs=[self.input_source_radio, self.video_input, self.webcam_input, self.model_dropdown],
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
                    self.input_source_radio,
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

            # Se o usuário selecionou a webcam, processar frames em tempo real
            self.webcam_input.stream(
                fn=self.on_webcam_frame,
                inputs=[
                    self.webcam_input,
                    self.detect_class_dropdown,
                    self.model_dropdown
                ],
                outputs=self.webcam_input,
            )

    def update_input_source_visibility(self, input_source):
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    def load_video_or_webcam(self, input_source, video_input, webcam_input, model_name):
        # Armazena o vídeo no estado e torna as opções visíveis
        detect_classes = self.get_detect_classes(model_name)
        if input_source == "Arquivo de Vídeo":
            input_data = video_input
        else:
            input_data = webcam_input
        return (
            input_data,
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

    def on_start_processing(self, input_source, input_data, detect_class, model_name):
        # Processa o vídeo ou webcam
        model_info = self.model_functions.get(model_name)
        if model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info["model_file"]

            if input_source == "Arquivo de Vídeo":
                # Processamento de arquivo de vídeo
                start_time = time.time()
                output_dir = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(output_dir, exist_ok=True)

                output_video_path, _ = start_processing_func(
                    input_data, output_dir, detect_class, model_file
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
                if os.path.exists(output_video_path_str):
                    file_size = os.path.getsize(output_video_path_str)
                    file_size_mb = file_size / (1024 * 1024)
                    file_size_str = f"{file_size_mb:.2f} MB"
                else:
                    file_size_str = "Arquivo não encontrado"

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
                # Processamento de webcam
                processing_message = "Processando vídeo da webcam..."
                file_info = ""
                return (
                    gr.update(visible=False),
                    "",
                    None,
                    gr.update(value=processing_message, visible=True),
                    gr.update(visible=False),
                    gr.update(value=file_info, visible=False),
                )
        else:
            return None, "Modelo não encontrado", None, None, None, None

    def on_webcam_frame(self, frame, detect_class, model_name):
        # Processa um frame da webcam
        model_info = self.model_functions.get(model_name)
        if model_info:
            process_webcam_frame_func = model_info["process_webcam_frame"]
            processed_frame = process_webcam_frame_func(frame, detect_class, model_info["model_file"])
            return processed_frame
        else:
            return frame

    def on_download_button_clicked(self, output_video_path):
        # Retorna o caminho do vídeo processado para download
        if os.path.exists(output_video_path):
            return output_video_path
        else:
            return None

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
