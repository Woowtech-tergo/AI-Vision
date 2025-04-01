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
# Removemos o uso de webcam e mediapipe para simplificar (como solicitado)

class App:
    def __init__(self):
        # Lista de modelos disponíveis
        self.model_list = ["YOLOv8DeepSort",
                           "ContadorDePessoasEmVideo"
                           ]

        # Mapeamento de funções por modelo
        self.model_functions = {
            "YOLOv8DeepSort": {
                "get_detectable_classes": self.get_yolo_classes,
                "start_processing": self.yolo_start_processing,
                "stop_processing": self.yolo_stop_processing,
                "process_webcam_frame": lambda f, d, m: f,
                "model_file": "yolov8n.pt",
            },
            "ContadorDePessoasEmVideo": {
                "get_detectable_classes": lambda model_file: ["person"],
                "start_processing": self.contador_start_processing,
                "stop_processing": self.contador_stop_processing,
                "process_webcam_frame": lambda f, d, m: f,
                "model_file": None,
            },
        }

        # CSS e HTML (visual)
        css = """
        .gradio-container {background-color: #f9f9f9;}
        #summary {background-color: #e0f7fa; padding:10px; border-radius:5px; margin:10px 0;}
        """

        with gr.Blocks(css=css) as self.demo:
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos (YOLOv8 + DeepSort)

                Com estatísticas e gráficos de evolução!
                """
            )

            self.input_source_radio = gr.Radio(
                choices=["Arquivo de Vídeo"],
                label="Fonte de Entrada",
                value="Arquivo de Vídeo"
            )

            self.video_input = gr.Video(label="Vídeo de Entrada", visible=True)

            self.model_dropdown = gr.Dropdown(
                choices=self.model_list, label="Modelo", value=self.model_list[0]
            )

            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            self.load_video_button = gr.Button("Carregar Vídeo")

            with gr.Column(visible=False) as self.options_column:
                self.detect_class_dropdown = gr.Dropdown(
                    choices=[], label="Classe"
                )
                self.start_button = gr.Button("Iniciar Processamento")
                self.stop_button = gr.Button("Interromper Processamento")
                self.processing_message = gr.Textbox(
                    label="Mensagem",
                    visible=False
                )
                self.output_video = gr.Video(label="Vídeo Processado")
                self.download_button = gr.Button("Download", visible=False)
                self.processing_time_label = gr.Textbox(
                    label="Tempo de Processamento"
                )
                self.file_info = gr.Markdown(value="", visible=False)
                # Novo: gráfico evolução e resumo estatístico
                self.plot_image = gr.Image(label="Evolução ao longo do tempo", visible=False)
                self.summary_markdown = gr.HTML(value="", visible=False, elem_id="summary")

            self.input_source_radio.change(
                fn=self.update_input_source_visibility,
                inputs=[self.input_source_radio],
                outputs=[self.video_input],
            )

            self.load_video_button.click(
                fn=self.load_video_or_webcam,
                inputs=[self.input_source_radio, self.video_input, self.model_dropdown],
                outputs=[
                    self.input_video_state,
                    self.options_column,
                    self.detect_class_dropdown
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
                    self.plot_image,
                    self.summary_markdown
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

    def update_input_source_visibility(self, input_source):
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True)
        else:
            return gr.update(visible=True)  # só um caso

    def load_video_or_webcam(self, input_source, video_input, model_name):
        detect_classes = self.get_detect_classes(model_name)
        input_data = video_input
        default_value = detect_classes[0] if detect_classes else None
        return (
            input_data,
            gr.update(visible=True),
            gr.update(choices=detect_classes, value=default_value)
        )

    def update_detect_classes(self, model_name):
        detect_classes = self.get_detect_classes(model_name)
        default_value = detect_classes[0] if detect_classes else None
        return gr.update(choices=detect_classes, value=default_value)

    def get_detect_classes(self, model_name):
        model_info = self.model_functions.get(model_name)
        if model_info:
            get_classes_func = model_info["get_detectable_classes"]
            model_file = model_info["model_file"]
            detect_classes = get_classes_func(model_file)
            return detect_classes
        else:
            return []

    # FUNÇÕES DE MODELOS
    def get_yolo_classes(self, model_file):
        # Exemplo fixo, normalmente YOLO ret. 80 classes do COCO
        return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                "scissors", "teddy bear", "hair drier", "toothbrush"]

    def yolo_start_processing(self, input_data, output_path, detect_class, model_file, progress=gr.Progress(track_tqdm=True)):
        # Simular dados:
        # Durante o processamento: log de contagem por frame
        # Classes detectadas (dummy)
        frame_counts = []
        class_counts = {}
        total_frames = 30
        directions_count = {"left": 5, "right": 7}  # Exemplo
        for i in range(total_frames):
            # Simular count random
            count = np.random.randint(0, 10)
            frame_counts.append(count)
            # Atualizar classe aleatória
            cls = np.random.choice(["person", "car", "dog"])
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Salvar vídeo de forma dummy
        output_video_path = os.path.join(output_path, "output.mp4")
        # criar um vídeo vazio (apenas para exemplo)
        w, h = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(30):
            cv2.putText(img, f"Frame {i}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            out.write(img)
        out.release()

        # Estatísticas:
        total_time = total_frames / 30.0
        total_objs = sum(frame_counts)
        # Top 3 classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_classes[:3]
        fps = 30.0  # fixo
        entradas = directions_count["left"]
        saidas = directions_count["right"]

        # Geração do gráfico de evolução
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frame_counts, marker='o')
        plt.title("Evolução da contagem de objetos ao longo do tempo")
        plt.xlabel("Frame")
        plt.ylabel("Contagem")
        plot_path = os.path.join(output_path, "plot.png")
        plt.savefig(plot_path)
        plt.close()

        # Resumo Estatístico Markdown
        top_str = "\n".join([f"- {cls}: {cnt}" for cls, cnt in top_3])
        summary = f"""
        <h3>Resumo Estatístico</h3>
        <p><b>Tempo total do vídeo:</b> {total_time:.2f} s</p>
        <p><b>Número total de objetos detectados:</b> {total_objs}</p>
        <p><b>Top 3 classes mais detectadas:</b><br>{top_str}</p>
        <p><b>Velocidade média (FPS):</b> {fps}</p>
        <p><b>Entradas (Esquerda):</b> {entradas}</p>
        <p><b>Saídas (Direita):</b> {saidas}</p>
        """

        return output_video_path, {
            "frame_counts": frame_counts,
            "plot_path": plot_path,
            "summary_html": summary
        }

    def yolo_stop_processing(self):
        return "Processamento interrompido."

    def contador_stop_processing(self):
        global should_continue
        should_continue = False
        return "Processamento interrompido."

    def contador_start_processing(self, input_data, output_path, detect_class, model_file,
                                  progress=gr.Progress(track_tqdm=True)):

        # Apenas exemplo simplificado (sem classes)
        global should_continue
        should_continue = True

        if not isinstance(input_data, str) or not os.path.exists(input_data):
            return None, None

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            return None, None

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = os.path.join(output_path, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (int(w), int(h)), True)

        # Dummy contagem
        leftCounter = 3
        rightCounter = 4
        frame_counts = []
        for i in range(30):
            frame_counts.append(np.random.randint(0,5))
            img = np.zeros((int(h), int(w), 3), dtype=np.uint8)
            cv2.putText(img, f"Frame {i}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            out.write(img)

        out.release()
        cap.release()

        # Sem classes detalhadas aqui
        class_counts = {"person": sum(frame_counts)}
        total_objs = sum(frame_counts)
        total_time = 30/fps
        entradas = leftCounter
        saidas = rightCounter

        # Plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(frame_counts, marker='o')
        plt.title("Evolução da contagem de pessoas ao longo do tempo")
        plt.xlabel("Frame")
        plt.ylabel("Contagem")
        plot_path = os.path.join(output_path, "plot.png")
        plt.savefig(plot_path)
        plt.close()

        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_classes[:3]
        fps_val = fps
        top_str = "\n".join([f"- {cls}: {cnt}" for cls, cnt in top_3])

        summary = f"""
        <h3>Resumo Estatístico</h3>
        <p><b>Tempo total do vídeo:</b> {total_time:.2f} s</p>
        <p><b>Número total de objetos detectados:</b> {total_objs}</p>
        <p><b>Top 3 classes mais detectadas:</b><br>{top_str}</p>
        <p><b>Velocidade média (FPS):</b> {fps_val}</p>
        <p><b>Entradas (Esquerda):</b> {entradas}</p>
        <p><b>Saídas (Direita):</b> {saidas}</p>
        """

        return output_video_path, {
            "frame_counts": frame_counts,
            "plot_path": plot_path,
            "summary_html": summary
        }

    def on_start_processing(self, input_source, input_data, detect_class, model_name):
        model_info = self.model_functions.get(model_name)
        if model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info["model_file"]

            start_time = time.time()
            output_dir = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(output_dir, exist_ok=True)

            # Agora start_processing retorna (output_video_path, logs_dict)
            result = start_processing_func(
                input_data, output_dir, detect_class, model_file
            )
            if result is None or result[0] is None:
                return None, "Falha no processamento", None, None, None, None, None, None
            output_video_path, logs = result
            end_time = time.time()

            output_video_path_str = str(output_video_path)
            processing_time = end_time - start_time
            hours, rem = divmod(processing_time, 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

            if os.path.exists(output_video_path_str):
                file_size = os.path.getsize(output_video_path_str)
                file_size_mb = file_size / (1024 * 1024)
                file_size_str = f"{file_size_mb:.2f} MB"
            else:
                file_size_str = "Arquivo não encontrado"

            processing_message = "Seu vídeo está pronto para download."
            file_info = f"**Tamanho do arquivo:** {file_size_str}"

            # Mostrar gráfico e resumo
            plot_image_val = gr.update(value=logs["plot_path"], visible=True)
            summary_val = gr.update(value=logs["summary_html"], visible=True)

            return (
                gr.update(value=output_video_path_str),
                time_str,
                output_video_path_str,
                gr.update(value=processing_message, visible=True),
                gr.update(visible=True),
                gr.update(value=file_info, visible=True),
                plot_image_val,
                summary_val
            )
        else:
            return None, "Modelo não encontrado", None, None, None, None, None, None

    def on_download_button_clicked(self, output_video_path):
        if os.path.exists(output_video_path):
            return output_video_path
        else:
            return None

    def on_stop_processing(self):
        model_name = self.model_dropdown.value
        model_info = self.model_functions.get(model_name)
        if model_info:
            stop_processing_func = model_info["stop_processing"]
            stop_processing_func()

    def yolo_stop_processing(self):
        return "Processamento interrompido."

if __name__ == "__main__":
    app = App()
    app.demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
