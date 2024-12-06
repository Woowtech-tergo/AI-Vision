
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
        self.model_list = ["YOLOv8DeepSort",
                           "ContadorDePessoasEmVideo",
                           "FaceMash"]  # Caso adicione outros modelos, inclua-os aqui


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

            "FaceMash": {
                "get_detectable_classes": lambda model_file: [],
                "start_processing": self.face_mash_start_processing,
                "stop_processing": self.face_mash_stop_processing,
                "process_webcam_frame": self.face_mash_process_webcam_frame,
                "model_file": None
            }

        }

        # Inicializa a interface Gradio
        with gr.Blocks() as self.demo:
            # Título
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos
                Baseado em OpenCV + YOLOv8 + DeepSort \n
                
                ContadorDePessoasEmVideo \n
                
                FaceMash
                
                
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
                self.webcam_input = WebRTC(
                    label="Webcam",

                    rtc_configuration={},

                )

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
                outputs=[self.webcam_input],
            )

    def update_input_source_visibility(self, input_source):
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    def load_video_or_webcam(self, input_source, video_input, webcam_input, model_name):
        detect_classes = self.get_detect_classes(model_name)
        input_data = video_input if input_source == "Arquivo de Vídeo" else webcam_input
        # Evitar IndexError caso detect_classes esteja vazio
        default_value = detect_classes[0] if detect_classes else None
        return (
            input_data,
            gr.update(visible=True),
            gr.update(choices=detect_classes, value=default_value)
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

    #========================================
    # Contador de Pessoas
    #========================================


    def contador_stop_processing(self):
        # Similar ao YOLOv8DeepSort, apenas ajusta a variável global se existir
        global should_continue
        should_continue = False
        return "Processamento interrompido."

    def contador_start_processing(self, input_data, output_path, detect_class, model_file,
                                  progress=gr.Progress(track_tqdm=True)):
        # Adapte o código do countingPeople.py aqui para:
        # - Ler o vídeo de input_data
        # - Processar frames e contar as pessoas
        # - Salvar o vídeo processado em output_path (por exemplo, output.mp4)
        # - Retornar (output_video_path, output_video_path)

        global should_continue
        should_continue = True

        # Verifica se input_data é um arquivo de vídeo (string) ou outro tipo
        if not isinstance(input_data, str) or not os.path.exists(input_data):
            return None, None

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {input_data}")
            return None, None

        # Ajustar parâmetros e linhas conforme countingPeople.py
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameArea = h * w
        areaTH = frameArea * 0.003

        leftmostLine = int(1.0 / 6 * w)
        rightmostLine = int(5.0 / 6 * w)

        leftmostLimit = int(1.0 / 12 * w)
        rightmostLimit = int(11.0 / 12 * w)

        leftmostLineColor = (255, 0, 0)
        rightmostLineColor = (0, 0, 255)

        # Linhas
        pt1 = [rightmostLine, 0]
        pt2 = [rightmostLine, h]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pt3 = [leftmostLine, 0]
        pt4 = [leftmostLine, h]
        pts_L2 = np.array([pt3, pt4], np.int32)

        pt5 = [leftmostLimit, 0]
        pt6 = [leftmostLimit, h]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pt7 = [rightmostLimit, 0]
        pt8 = [rightmostLimit, h]
        pts_L4 = np.array([pt7, pt8], np.int32)

        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
        kernelOp = np.ones((3, 3), np.uint8)
        kernelCl = np.ones((9, 9), np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        persons = []
        max_p_age = 5
        pid = 1

        leftCounter = 0
        rightCounter = 0

        # Configura saída de vídeo
        output_video_path = os.path.join(output_path, "output.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(w), int(h))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, size, True)

        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                break

            for per in persons:
                per.age_one()

            fgmask = backgroundSubtractor.apply(frame)
            ret2, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > areaTH:
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w_box, h_box = cv2.boundingRect(cnt)

                    newPerson = True
                    if cy in range(leftmostLimit, rightmostLimit):
                        for person in persons:
                            if abs(x - person.getX()) <= w_box and abs(y - person.getY()) <= h_box:
                                newPerson = False
                                person.updateCoords(cx, cy)

                                if person.goingLeft(rightmostLine, leftmostLine) == True:
                                    leftCounter += 1
                                elif person.goingRight(rightmostLine, leftmostLine) == True:
                                    rightCounter += 1
                                break

                            if person.getState() == '1':
                                if person.getDir() == 'right' and person.getY() > rightmostLimit:
                                    person.setDone()
                                elif person.getDir() == 'left' and person.getY() < leftmostLimit:
                                    person.setDone()

                            if person.timedOut():
                                index = persons.index(person)
                                persons.pop(index)
                                del person

                        if newPerson == True:
                            from Modelos.ContadorDePessoasEmVideo.Person import Person
                            person = Person(pid, cx, cy, max_p_age)
                            persons.append(person)
                            pid += 1

                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        img = cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

            leftMsg = 'Left: ' + str(leftCounter)
            rightMsg = 'Right: ' + str(rightCounter)
            frame = cv2.polylines(frame, [pts_L1], False, rightmostLineColor, thickness=2)
            frame = cv2.polylines(frame, [pts_L2], False, leftmostLineColor, thickness=2)
            frame = cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
            frame = cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
            cv2.putText(frame, leftMsg, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, leftMsg, (10, 40), font, 0.5, leftmostLineColor, 1, cv2.LINE_AA)
            cv2.putText(frame, rightMsg, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, rightMsg, (10, 90), font, 0.5, rightmostLineColor, 1, cv2.LINE_AA)

            out.write(frame)

        cap.release()
        out.release()

        return output_video_path, output_video_path

    def contador_process_webcam_frame(self, frame, detect_class, model_file):
        # Caso queira processar frame a frame da webcam com o mesmo contador, é preciso adaptar o código
        # do countingPeople para trabalhar frame a frame sem limites.
        # Por simplicidade, retornaremos o frame original ou algum processamento mínimo.
        return frame

    # ========================================
    # Funções necessárias para FaceMash:
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

        import mediapipe as mp
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