
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
# import mediapipe as mp

# testing git
class App:
    def __init__(self):
        # Lista de modelos disponíveis
        self.model_list = ["YOLOv8DeepSort",
                           "ContadorDePessoasEmVideo"
                           ]



        # Mapeamento de funções por modelo
        self.model_functions = {
            # Adicione outros modelos aqui
            "YOLOv8DeepSort": {
                "get_detectable_classes": yolo_deepsort.get_detectable_classes,
                "start_processing": yolo_deepsort.start_processing,
                "stop_processing": yolo_deepsort.stop_processing,
                "process_webcam_frame": yolo_deepsort.process_webcam_frame,
                "model_file": "yolov8n.pt",  # Modelo padrão para YOLOv8DeepSort
            },

            "ContadorDePessoasEmVideo": {
                "get_detectable_classes": lambda model_file: ["person"],  # Pode ser ["person"] ou []
                "start_processing": self.contador_start_processing,
                "stop_processing": self.contador_stop_processing,
                "process_webcam_frame": self.contador_process_webcam_frame,
                "model_file": None,  # Não há arquivo de modelo específico
            },

            # "FaceMash": {
            #     "get_detectable_classes": lambda model_file: [],
            #     "start_processing": self.face_mash_start_processing,
            #     "stop_processing": self.face_mash_stop_processing,
            #     "process_webcam_frame": self.face_mash_process_webcam_frame,
            #     "model_file": None
            # }

        }

        # # Inicializa a interface Gradio
        with gr.Blocks(
                css=".gradio-container {background-color: #f000000; padding: 20px;}"

        ) as self.demo:
            gr.HTML("<h2 style='color:blue;'> Projeto de Contador usando Visão Computacional</h2>")

            # Título
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos
                Modelo 1: Baseado em OpenCV + YOLOv8 + DeepSort \n

                Modelo 2: ContadorDePessoasEmVideo \n

                """
            )

            # Selecionar fonte de entrada
            with gr.Row():
                self.input_source_radio = gr.Radio(
                    choices=["Arquivo de Vídeo"],  # Webcam
                    label="Fonte de Entrada",
                    value="Arquivo de Vídeo"
                )

            # Carregar vídeo ou usar webcam
            with gr.Row():
                self.video_input = gr.Video(label="Vídeo de Entrada", visible=True)
                # self.webcam_input = WebRTC(
                #     label="Webcam",
                #
                #     rtc_configuration={},
                #
                # )

            # Selecionar modelo
            with gr.Row():
                self.model_dropdown = gr.Dropdown(
                    choices=self.model_list, label="Modelo", value=self.model_list[0]
                )
        #
            # Estados para armazenar o vídeo de entrada e o caminho do vídeo processado
            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            # Botão para carregar o vídeo ou iniciar a webcam
            with gr.Row():
                self.load_video_button = gr.Button("Carregar Vídeo")
        #
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

            # Eventos - output opcional:  self.webcam_input
            self.input_source_radio.change(
                fn=self.update_input_source_visibility,
                inputs=[self.input_source_radio],
                outputs=[self.video_input],
            )

            # Parametro Opcional no Input: self.webcam_input
            self.load_video_button.click(
                fn=self.load_video_or_webcam,
                inputs=[self.input_source_radio, self.video_input, self.model_dropdown],
                outputs=[
                    self.input_video_state,  # recebe input_data
                    self.options_column,  # recebe gr.update(visible=True)
                    self.detect_class_dropdown  # recebe gr.update(choices=..., value=...)
                ],
            )
        #
        #     self.model_dropdown.change(
        #         fn=self.update_detect_classes,
        #         inputs=[self.model_dropdown],
        #         outputs=[self.detect_class_dropdown],
        #     )
        #
        #     self.start_button.click(
        #         fn=self.on_start_processing,
        #         inputs=[
        #             self.input_source_radio,
        #             self.input_video_state,
        #             self.detect_class_dropdown,
        #             self.model_dropdown,
        #         ],
        #         outputs=[
        #             self.output_video,
        #             self.processing_time_label,
        #             self.output_video_path_state,
        #             self.processing_message,
        #             self.download_button,
        #             self.file_info,
        #         ],
        #     )
        #
        #     self.download_button.click(
        #         fn=self.on_download_button_clicked,
        #         inputs=[self.output_video_path_state],
        #         outputs=gr.File(file_count="single"),
        #     )
        #
        #     self.stop_button.click(
        #         fn=self.on_stop_processing,
        #         inputs=[],
        #         outputs=[],
        #     )
        #
        #     # Se o usuário selecionou a webcam, processar frames em tempo real
        #     # self.webcam_input.stream(
        #     #     fn=self.on_webcam_frame,
        #     #     inputs=[
        #     #         self.webcam_input,
        #     #         self.detect_class_dropdown,
        #     #         self.model_dropdown
        #     #     ],
        #     #     outputs=[self.webcam_input],
        #     # )

    def update_input_source_visibility(self, input_source):
        # Como "Arquivo de Vídeo" é a única opção e o valor padrão,
        # esta função pode simplesmente garantir que o input de vídeo esteja visível.
        # Ou, se você quiser manter a lógica caso adicione a webcam de volta:
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True)  # Retorna apenas UM update para video_input
        # else: # Se a webcam fosse uma opção
        # return gr.update(visible=False)
        # Dado o estado atual (apenas vídeo), podemos simplificar para:
        return gr.update(visible=True)

    # Adicionar webcam_input quando necessario
    def load_video_or_webcam(self, input_source, video_input,  model_name):
        detect_classes = self.get_detect_classes(model_name)
        input_data = video_input if input_source == "Arquivo de Vídeo" else None # webcam_input
        # Evitar IndexError caso detect_classes esteja vazio
        detect_classes = self.get_detect_classes(model_name)
        default_value = detect_classes[0] if detect_classes else None

        return (
            input_data,  # para self.input_video_state (um State, pode receber um valor simples)
            gr.update(visible=True),  # para self.options_column (um Column, aceita visible)
            gr.update(choices=detect_classes, value=default_value)  # para o Dropdown (accepta choices e value)
        )

    def update_detect_classes(self, model_name):
        detect_classes = self.get_detect_classes(model_name)
        # Evitar IndexError caso detect_classes esteja vazio
        default_value = detect_classes[0] if detect_classes else None
        return gr.update(choices=detect_classes, value=default_value)

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
        model_info = self.model_functions.get(model_name)
        if model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info["model_file"]

            # Como agora só existe a opção "Arquivo de Vídeo"
            start_time = time.time()
            output_dir = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(output_dir, exist_ok=True)

            output_video_path, _ = start_processing_func(
                input_data, output_dir, detect_class, model_file
            )

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

            return (
                gr.update(value=output_video_path_str),  # self.output_video atualiza com o caminho do vídeo
                time_str,  # self.processing_time_label
                output_video_path_str,  # self.output_video_path_state
                gr.update(value=processing_message, visible=True),  # self.processing_message
                gr.update(visible=True),  # self.download_button
                gr.update(value=file_info, visible=True),  # self.file_info
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

    #========================================
    # Contador de Pessoas
    #========================================


    def contador_stop_processing(self):
        # Similar ao YOLOv8DeepSort, apenas ajusta a variável global se existir
        global should_continue
        should_continue = False
        return "Processamento interrompido."

        # ANTES:
        # def contador_start_processing(self, input_data, output_path, detect_class, model_file,
        #                               progress=gr.Progress(track_tqdm=True)):

        # DEPOIS (PARA TESTE):

    def contador_start_processing(self, input_data, output_path, detect_class, model_file):
        # Declaração global e atribuição inicial (UMA VEZ)
        global should_continue
        should_continue = True

        # Verifica se input_data é um arquivo de vídeo (string) ou outro tipo
        if not isinstance(input_data, str) or not os.path.exists(input_data):
            print(f"Erro: input_data não é um caminho de vídeo válido: {input_data}")
            return None, None # Retorna cedo

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {input_data}")
            return None, None # Retorna cedo

        # Ajustar parâmetros e linhas conforme countingPeople.py
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # Adicionar verificação para w, h > 0 seria bom aqui
        frameArea = h * w
        areaTH = frameArea * 0.003

        # ... (resto da configuração de linhas, cores, etc.) ...

        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
        # ... (kernels, fontes, listas, etc.) ...
        persons = []
        pid = 1
        leftCounter = 0
        rightCounter = 0

        # Configura saída de vídeo
        output_video_path = os.path.join(output_path, "output.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Adicionar verificação para fps > 0 seria bom aqui
        size = (int(w), int(h))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, size, True)
        # Adicionar verificação se out.isOpened() seria bom aqui

        # Loop principal de processamento
        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                break # Sai do loop

            # ... (Toda a lógica de processamento do frame:
            #      envelhecer pessoas, aplicar máscara, contornos,
            #      tracking, desenhar na imagem, etc.) ...

            out.write(frame) # Escreve o frame processado

        # Limpeza após o loop
        print("Fim do processamento, liberando recursos.")
        cap.release()
        out.release()

        # Retorno final (UMA VEZ, no fim da função)
        return output_video_path, output_video_path


    def contador_process_webcam_frame(self, frame, detect_class, model_file):
        """" Caso queira processar frame a frame da webcam com o mesmo contador, é preciso adaptar o código
        do countingPeople para trabalhar frame a frame sem limites."""        
        return frame

    # ========================================
    # DESATIVADAS - Funções necessárias para FaceMash:
    # ========================================

    def face_mash_stop_processing(self):
        global should_continue
        should_continue = False
        return "Processamento interrompido."

    def face_mash_start_processing(self, input_data, output_path, detect_class, model_file,
                                   progress=gr.Progress(track_tqdm=True)):
        global should_continue
        should_continue = True

        # Adaptado para vídeo fornecido pelo usuário (input_data é o caminho do vídeo)
        if not isinstance(input_data, str) or not os.path.exists(input_data):
            return None, None

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {input_data}")
            return None, None

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        output_video_path = os.path.join(output_path, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(w), int(h)), True)


        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
            while cap.isOpened() and should_continue:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saida_facemesh = facemesh.process(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if saida_facemesh.multi_face_landmarks:
                    for face_landmarks in saida_facemesh.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            face_landmarks,
                            mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1,
                                                                         circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1,
                                                                           circle_radius=1)
                        )

                out.write(frame_bgr)

        cap.release()
        out.release()
        return output_video_path, output_video_path

    def face_mash_process_webcam_frame(self, frame, detect_class, model_file):
        # Caso no futuro queira usar webcam, já está pronto. Mas com vídeo, não é chamado.
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
            saida_facemesh = facemesh.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if saida_facemesh.multi_face_landmarks:
                for face_landmarks in saida_facemesh.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1,
                                                                     circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1,
                                                                       circle_radius=1)
                    )

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    app = App()
    app.demo.launch(server_name="0.0.0.0", server_port=7860, share=True)