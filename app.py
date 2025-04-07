# app.py

from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import gradio as gr

# Se você tiver um módulo próprio "yolo_deepsort.py", importe as funções específicas
import deep_sort.deep_sort.deep_sort as ds
from Modelos.YOLOv8DeepSortTracking import yolo_deepsort

# Variável global para interrupção dos loops de processamento
should_continue = True


class App:
    def __init__(self):
        # Lista de modelos disponíveis
        self.model_list = [
            "YOLOv8DeepSort",
            "ContadorDePessoasEmVideo",
            # "FaceMash"  # Se quiser reativar
        ]

        # Mapeamento de funções por modelo
        self.model_functions = {
            "YOLOv8DeepSort": {
                "get_detectable_classes": yolo_deepsort.get_detectable_classes,
                "start_processing": yolo_deepsort.start_processing,
                "stop_processing": yolo_deepsort.stop_processing,
                "process_webcam_frame": yolo_deepsort.process_webcam_frame,
                "model_file": "yolov8n.pt",  # Modelo default YOLOv8
            },
            "ContadorDePessoasEmVideo": {
                "get_detectable_classes": lambda model_file: ["person"],  # Ex: só "person"
                "start_processing": self.contador_start_processing,
                "stop_processing": self.contador_stop_processing,
                "process_webcam_frame": self.contador_process_webcam_frame,
                "model_file": None,  # Não há arquivo .pt específico
            },
            # Se quiser adicionar FaceMash, reative as funções
            # "FaceMash": {
            #     "get_detectable_classes": lambda model_file: [],
            #     "start_processing": self.face_mash_start_processing,
            #     "stop_processing": self.face_mash_stop_processing,
            #     "process_webcam_frame": self.face_mash_process_webcam_frame,
            #     "model_file": None
            # }
        }

        # Construção da UI com Blocks
        with gr.Blocks() as self.demo:
            gr.HTML("<h2 style='color:blue;'>Projeto de Contador usando Visão Computacional</h2>")

            gr.Markdown(
                """
                **Detecção e Rastreamento de Objetos**  

                - Modelo 1: Baseado em OpenCV + YOLOv8 + DeepSort  
                - Modelo 2: ContadorDePessoasEmVideo  
                """
            )

            # Fonte de entrada de vídeo (por enquanto só arquivo)
            with gr.Row():
                self.input_source_radio = gr.Radio(
                    choices=["Arquivo de Vídeo"],
                    label="Fonte de Entrada",
                    value="Arquivo de Vídeo"
                )

            # Carregar vídeo
            with gr.Row():
                self.video_input = gr.Video(label="Vídeo de Entrada", visible=True)

            # Selecionar modelo
            with gr.Row():
                self.model_dropdown = gr.Dropdown(
                    choices=self.model_list, label="Modelo", value=self.model_list[0]
                )

            # Estados do Gradio para armazenar caminhos temporários
            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            # Botão para carregar o vídeo
            with gr.Row():
                self.load_video_button = gr.Button("Carregar Vídeo")

            # Opções do processamento (visíveis só depois de carregar vídeo)
            with gr.Column(visible=False) as self.options_column:
                with gr.Row():
                    self.detect_class_dropdown = gr.Dropdown(
                        choices=[], label="Classe a Detectar"
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

            # ----- Definições dos eventos -----

            # Controle de visibilidade do input de vídeo, caso no futuro sejam adicionadas outras fontes.
            self.input_source_radio.change(
                fn=self.update_input_source_visibility,
                inputs=[self.input_source_radio],
                outputs=[self.video_input],
            )

            # Carregar o arquivo de vídeo e disponibilizar as opções
            self.load_video_button.click(
                fn=self.load_video_or_webcam,
                inputs=[self.input_source_radio, self.video_input, self.model_dropdown],
                outputs=[
                    self.input_video_state,
                    self.options_column,
                    self.detect_class_dropdown
                ],
            )

            # Atualiza dropdown de classes quando troca de modelo
            self.model_dropdown.change(
                fn=self.update_detect_classes,
                inputs=[self.model_dropdown],
                outputs=[self.detect_class_dropdown],
            )

            # Inicia processamento
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

            # Faz download do vídeo
            self.download_button.click(
                fn=self.on_download_button_clicked,
                inputs=[self.output_video_path_state],
                outputs=gr.File(file_count="single"),
            )

            # Interrompe processamento
            self.stop_button.click(
                fn=self.on_stop_processing,
                inputs=[],
                outputs=[],
            )

    def update_input_source_visibility(self, input_source):
        """
        Se no futuro quiser adicionar "Webcam", atualize esta função.
        Por enquanto, só mantemos o vídeo visível.
        """
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True)
        return gr.update(visible=False)

    def load_video_or_webcam(self, input_source, video_input, model_name):
        """
        Função chamada ao clicar no botão "Carregar Vídeo".
        Retorna:
          - Estado interno com o caminho do vídeo
          - Torna visíveis as opções de processamento
          - Atualiza dropdown de classes
        """
        detect_classes = self.get_detect_classes(model_name)

        # Se não houver classes, não setamos valor default
        default_value = detect_classes[0] if detect_classes else None

        # input_data será o arquivo de vídeo
        input_data = video_input

        return (
            input_data,
            gr.update(visible=True),
            gr.update(choices=detect_classes, value=default_value)
        )

    def update_detect_classes(self, model_name):
        """
        Atualiza as classes detectáveis de acordo com o modelo selecionado.
        """
        detect_classes = self.get_detect_classes(model_name)
        default_value = detect_classes[0] if detect_classes else None
        return gr.update(choices=detect_classes, value=default_value)

    def get_detect_classes(self, model_name):
        """
        Obtém as classes detectáveis do modelo selecionado.
        """
        model_info = self.model_functions.get(model_name)
        if model_info:
            get_classes_func = model_info["get_detectable_classes"]
            model_file = model_info["model_file"]
            detect_classes = get_classes_func(model_file)
            return detect_classes
        else:
            return []

    def on_start_processing(self, input_source, input_data, detect_class, model_name):
        """
        Aciona a função de processamento do modelo escolhido.
        """
        model_info = self.model_functions.get(model_name)
        if model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info["model_file"]

            start_time = time.time()
            output_dir = os.path.join(os.getcwd(), 'outputs')
            os.makedirs(output_dir, exist_ok=True)

            # Chamamos a função do modelo selecionado
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
                gr.update(value=output_video_path_str),
                time_str,
                output_video_path_str,
                gr.update(value=processing_message, visible=True),
                gr.update(visible=True),
                gr.update(value=file_info, visible=True),
            )
        else:
            # Caso não encontre o modelo
            return (
                None,
                "Modelo não encontrado",
                None,
                gr.update(value="Erro ao iniciar.", visible=True),
                gr.update(visible=False),
                gr.update(value="", visible=False),
            )

    def on_download_button_clicked(self, output_video_path):
        """
        Retorna o caminho do arquivo para que o componente File possa disponibilizá-lo.
        """
        if os.path.exists(output_video_path):
            return output_video_path
        else:
            return None

    def on_stop_processing(self):
        """
        Chama a função de interrupção do modelo selecionado.
        """
        model_name = self.model_dropdown.value
        model_info = self.model_functions.get(model_name)
        if model_info:
            stop_processing_func = model_info["stop_processing"]
            stop_processing_func()
        return

    # ==========================
    #  Contador de Pessoas
    # ==========================
    def contador_start_processing(self, input_data, output_path, detect_class, model_file):
        global should_continue
        should_continue = True

        if not isinstance(input_data, str) or not os.path.exists(input_data):
            print(f"Erro: input_data não é um caminho de vídeo válido: {input_data}")
            return None, None

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {input_data}")
            return None, None

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Diretorio de saída
        os.makedirs(output_path, exist_ok=True)
        output_video_path = os.path.join(output_path, "output.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(w), int(h)), True)

        # Exemplo bem simplificado de contagem
        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
        count_left = 0
        count_right = 0

        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                break

            # Exemplo de uso do subtrator e contagem mínima
            fgmask = backgroundSubtractor.apply(frame)
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                # Este é só um exemplo, não é uma contagem real robusta
                if w_box * h_box > 400:  # threshold de área
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    # Log de contagem fictício
                    if x < frame.shape[1] // 2:
                        count_left += 1
                    else:
                        count_right += 1

            cv2.putText(frame, f"Left: {count_left} | Right: {count_right}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame)

        cap.release()
        out.release()

        return output_video_path, output_video_path

    def contador_stop_processing(self):
        global should_continue
        should_continue = False
        return "Processamento interrompido."

    def contador_process_webcam_frame(self, frame, detect_class, model_file):
        """
        Se fosse usar webcam, precisaria de lógica frame-a-frame aqui.
        Mantém vazio ou genérico para evitar conflito de schema.
        """
        return frame

    # ==========================
    #  (Opcional) FaceMash
    # ==========================
    # Se quiser reativar, lembre-se de remover parâmetros extras de função
    # e comentar qualquer coisa que possa gerar conflito no schema.

    # def face_mash_start_processing(self, input_data, output_path, detect_class, model_file):
    #     global should_continue
    #     should_continue = True
    #     import mediapipe as mp
    #
    #     if not isinstance(input_data, str) or not os.path.exists(input_data):
    #         return None, None
    #
    #     cap = cv2.VideoCapture(input_data)
    #     if not cap.isOpened():
    #         print(f"Não foi possível abrir o vídeo: {input_data}")
    #         return None, None
    #
    #     w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     output_video_path = os.path.join(output_path, "output.mp4")
    #
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(w), int(h)), True)
    #
    #     mp_drawing = mp.solutions.drawing_utils
    #     mp_face_mesh = mp.solutions.face_mesh
    #
    #     with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    #         while cap.isOpened() and should_continue:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break
    #
    #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             saida_facemesh = facemesh.process(frame_rgb)
    #             frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    #
    #             if saida_facemesh.multi_face_landmarks:
    #                 for face_landmarks in saida_facemesh.multi_face_landmarks:
    #                     mp_drawing.draw_landmarks(
    #                         frame_bgr,
    #                         face_landmarks,
    #                         mp_face_mesh.FACEMESH_CONTOURS
    #                     )
    #             out.write(frame_bgr)
    #
    #     cap.release()
    #     out.release()
    #     return output_video_path, output_video_path
    #
    # def face_mash_stop_processing(self):
    #     global should_continue
    #     should_continue = False
    #     return "Processamento interrompido."
    #
    # def face_mash_process_webcam_frame(self, frame, detect_class, model_file):
    #     import mediapipe as mp
    #     mp_drawing = mp.solutions.drawing_utils
    #     mp_face_mesh = mp.solutions.face_mesh
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    #         saida_facemesh = facemesh.process(frame_rgb)
    #         frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    #         if saida_facemesh.multi_face_landmarks:
    #             for face_landmarks in saida_facemesh.multi_face_landmarks:
    #                 mp_drawing.draw_landmarks(
    #                     frame_bgr,
    #                     face_landmarks,
    #                     mp_face_mesh.FACEMESH_CONTOURS
    #                 )
    #     return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    app = App()
    # Ajuste a porta se precisar, mas Railway normalmente usa PORT=8000 (ou outra).
    # Você pode usar os "Railway env" pra detectar a porta.
    # Exemplo de detect:
    # import os
    # port = int(os.environ.get("PORT", 7860))
    # app.demo.launch(server_name="0.0.0.0", server_port=port)
    app.demo.launch(server_name="0.0.0.0", server_port=7860)
