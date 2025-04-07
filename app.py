# app.py

from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
import time
import os

# Verifique se este import está correto e se a estrutura de pastas está acessível no Railway
# Exemplo: Se deep_sort está no mesmo nível que app.py, o import pode precisar ser ajustado
# dependendo de como você estruturou o módulo.
# Se 'deep_sort' é uma pasta no mesmo nível de app.py contendo outra pasta 'deep_sort' com o arquivo .py:
# from deep_sort.deep_sort import deep_sort as ds # Parece correto se a estrutura for deep_sort/deep_sort/deep_sort.py
# Se for deep_sort/deep_sort.py:
# from deep_sort import deep_sort as ds
# Verifique a estrutura exata no seu repositório. Assumindo que o original está correto:
import deep_sort.deep_sort.deep_sort as ds


import gradio as gr
# Tente remover a importação de WebRTC se não estiver usando para simplificar
# from gradio_webrtc import WebRTC # Comentado se não for usar webcam

# Importar o módulo yolo_deepsort completo
# Verifique a estrutura de pastas para 'Modelos' também
from Modelos.YOLOv8DeepSortTracking import yolo_deepsort
from Modelos.ContadorDePessoasEmVideo.Person import Person # Import Person aqui se necessário globalmente ou mova para dentro da função

# Váriavel global para controle de parada (Defina fora da classe ou passe como argumento)
should_continue = True


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
                "process_webcam_frame": self.contador_process_webcam_frame, # Função placeholder
                "model_file": None,  # Não há arquivo de modelo específico
            },
        }

        # Inicializa a interface Gradio
        with gr.Blocks(
                css=".gradio-container {background-color: #f0f0f0; padding: 20px;}" # Ajustei a cor de fundo

        ) as self.demo:
            gr.HTML("<h2 style='color:blue;'>AI'Vision - Contador e Detector</h2>") # Título atualizado

            # Título
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos
                **Modelo 1:** Baseado em OpenCV + YOLOv8 + DeepSort \n
                **Modelo 2:** Contador De Pessoas Em Video (OpenCV Background Subtraction) \n
                ---
                """
            )

            # Selecionar fonte de entrada
            with gr.Row():
                self.input_source_radio = gr.Radio(
                    choices=["Arquivo de Vídeo"],
                    label="Fonte de Entrada",
                    value="Arquivo de Vídeo"
                )

            # Carregar vídeo ou usar webcam
            with gr.Row():
                self.video_input = gr.Video(label="Vídeo de Entrada", visible=True)
                # Webcam comentada por enquanto
                # self.webcam_input = WebRTC(...)

            # Selecionar modelo
            with gr.Row():
                self.model_dropdown = gr.Dropdown(
                    choices=self.model_list, label="Modelo", value=self.model_list[0]
                )

            # Estados para armazenar o vídeo de entrada e o caminho do vídeo processado
            self.input_video_state = gr.State()
            self.output_video_path_state = gr.State()

            # Botão para carregar o vídeo (ativa as opções)
            with gr.Row():
                self.load_video_button = gr.Button("Carregar Vídeo e Mostrar Opções")

            # Após carregar o vídeo ou iniciar a webcam, mostram-se as opções adicionais
            with gr.Column(visible=False) as self.options_column:
                gr.Markdown("### Opções de Processamento")
                with gr.Row():
                    self.detect_class_dropdown = gr.Dropdown(
                        choices=[], label="Classe para Detecção (se aplicável)"
                    )
                with gr.Row():
                    self.start_button = gr.Button("Iniciar Processamento") # Botão ainda existe, mas handler está comentado
                    self.stop_button = gr.Button("Interromper Processamento")
                with gr.Row():
                    self.processing_message = gr.Textbox(
                        label="Status",
                        visible=False,
                        interactive=False
                    )
                with gr.Row():
                    self.output_video = gr.Video(label="Vídeo Processado", interactive=False)
                with gr.Row():
                    self.download_button = gr.Button("Download Vídeo Processado", visible=False)
                    self.processing_time_label = gr.Textbox(
                        label="Tempo de Processamento",
                        interactive=False
                    )
                    self.file_info = gr.Markdown(
                        value="", visible=False
                    )

            # --- Event Handlers ---

            # Handler problemático/redundante COMENTADO
            # self.input_source_radio.change(
            #     fn=self.update_input_source_visibility,
            #     inputs=[self.input_source_radio],
            #     outputs=[self.video_input],
            # )

            # Handler COMENTADO (pode ser descomentado após confirmar que a app inicia)
            # self.load_video_button.click(
            #     fn=self.load_video_or_webcam,
            #     inputs=[self.input_source_radio, self.video_input, self.model_dropdown],
            #     outputs=[
            #         self.input_video_state,
            #         self.options_column,
            #         self.detect_class_dropdown
            #     ],
            # )

            # Handler OK (Atualiza classes quando modelo muda)
            self.model_dropdown.change(
                fn=self.update_detect_classes,
                inputs=[self.model_dropdown],
                outputs=[self.detect_class_dropdown],
            )

            # Handler do botão Iniciar COMENTADO (Principal suspeito do erro)
            # self.start_button.click(
            #     fn=self.on_start_processing,
            #     inputs=[
            #         self.input_source_radio,
            #         self.input_video_state,
            #         self.detect_class_dropdown,
            #         self.model_dropdown,
            #     ],
            #     outputs=[
            #         self.output_video,
            #         self.processing_time_label,
            #         self.output_video_path_state,
            #         self.processing_message,
            #         self.download_button,
            #         self.file_info,
            #     ],
            # )

            # Handler OK (Download)
            self.download_button.click(
                fn=self.on_download_button_clicked,
                inputs=[self.output_video_path_state],
                outputs=gr.File(file_count="single"),
            )

            # Handler OK (Stop)
            self.stop_button.click(
                fn=self.on_stop_processing,
                inputs=[], # Nenhum input direto da UI
                outputs=[self.processing_message], # Atualiza a mensagem
            )

    # --- Métodos da Classe App ---

    def update_input_source_visibility(self, input_source):
        # Esta função não é mais chamada se o handler .change estiver comentado
        if input_source == "Arquivo de Vídeo":
            return gr.update(visible=True)
        return gr.update(visible=True)

    def load_video_or_webcam(self, input_source, video_input_path, model_name):
        # Esta função não é mais chamada se o handler .click do load_video_button estiver comentado
        # Se for descomentar, video_input é o *caminho* do arquivo temporário
        detect_classes = self.get_detect_classes(model_name)
        default_value = detect_classes[0] if detect_classes else None

        # Atualiza o state, torna opções visíveis, atualiza dropdown de classes
        return (
            video_input_path, # Salva o caminho do vídeo no state
            gr.update(visible=True), # Torna a coluna de opções visível
            gr.update(choices=detect_classes, value=default_value) # Atualiza o dropdown
        )

    def update_detect_classes(self, model_name):
        detect_classes = self.get_detect_classes(model_name)
        default_value = detect_classes[0] if detect_classes else None
        # Atualiza apenas o dropdown de classes
        return gr.update(choices=detect_classes, value=default_value)

    def get_detect_classes(self, model_name):
        model_info = self.model_functions.get(model_name)
        if model_info and "get_detectable_classes" in model_info:
            get_classes_func = model_info["get_detectable_classes"]
            model_file = model_info.get("model_file") # Use .get para evitar erro se 'model_file' não existir
            try:
                detect_classes = get_classes_func(model_file)
                # Garante que sempre retorna uma lista
                return detect_classes if isinstance(detect_classes, list) else []
            except Exception as e:
                print(f"Erro ao obter classes para {model_name}: {e}")
                return []
        else:
            print(f"Modelo ou função get_detectable_classes não encontrado para: {model_name}")
            return []

    def on_start_processing(self, input_source, input_video_path_state, detect_class, model_name):
        # Esta função não será chamada se o handler .click do start_button estiver comentado
        model_info = self.model_functions.get(model_name)
        if model_info and "start_processing" in model_info:
            start_processing_func = model_info["start_processing"]
            model_file = model_info.get("model_file")

            if input_source == "Arquivo de Vídeo" and input_video_path_state:
                start_time = time.time()
                # Define diretório de saída dentro do diretório de trabalho atual
                output_dir = os.path.join(os.getcwd(), 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                print(f"Iniciando processamento para: {input_video_path_state}")

                try:
                    # Chama a função específica do modelo
                    output_video_path, _ = start_processing_func(
                        input_video_path_state, output_dir, detect_class, model_file
                    )

                    # Verifica se o processamento retornou um caminho válido
                    if output_video_path and os.path.exists(output_video_path):
                        end_time = time.time()
                        output_video_path_str = str(output_video_path)

                        processing_time = end_time - start_time
                        hours, rem = divmod(processing_time, 3600)
                        minutes, seconds = divmod(rem, 60)
                        time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

                        file_size = os.path.getsize(output_video_path_str)
                        file_size_mb = file_size / (1024 * 1024)
                        file_size_str = f"{file_size_mb:.2f} MB"

                        processing_message = "Processamento concluído!"
                        file_info = f"**Arquivo:** `{os.path.basename(output_video_path_str)}`\n**Tamanho:** {file_size_str}"

                        print(f"Processamento concluído: {output_video_path_str}")

                        # Retorna os 6 valores esperados pelo handler .click
                        return (
                            gr.update(value=output_video_path_str, interactive=False), # Atualiza o vídeo de saída
                            time_str,  # Atualiza o tempo (string vai direto pro Textbox)
                            output_video_path_str,  # Atualiza o state com o caminho
                            gr.update(value=processing_message, visible=True), # Atualiza a mensagem
                            gr.update(visible=True),  # Torna o botão de download visível
                            gr.update(value=file_info, visible=True),  # Mostra informações do arquivo
                        )
                    else:
                        print("Erro: Função de processamento não retornou um caminho de vídeo válido.")
                        return None, "Erro no processamento", None, gr.update(value="Erro ao gerar vídeo.", visible=True), None, None

                except Exception as e:
                    print(f"Erro durante start_processing para {model_name}: {e}")
                    # Retorna mensagens de erro para a interface
                    error_message = f"Erro ao processar: {e}"
                    return None, "Erro", None, gr.update(value=error_message, visible=True), gr.update(visible=False), gr.update(visible=False)

            else:
                # Caso input_source não seja Arquivo de Vídeo ou state esteja vazio
                message = "Nenhum arquivo de vídeo carregado."
                print(message)
                return None, "", None, gr.update(value=message, visible=True), gr.update(visible=False), gr.update(visible=False)
        else:
            message = f"Função start_processing não encontrada para o modelo: {model_name}"
            print(message)
            return None, "Modelo inválido", None, gr.update(value=message, visible=True), gr.update(visible=False), gr.update(visible=False)


    def on_webcam_frame(self, frame, detect_class, model_name):
        # Função placeholder, não chamada atualmente
        return frame

    def on_download_button_clicked(self, output_video_path):
        if output_video_path and os.path.exists(output_video_path):
            print(f"Preparando download para: {output_video_path}")
            return output_video_path
        else:
            print("Erro: Caminho para download inválido ou arquivo não existe.")
            # Talvez retornar uma mensagem de erro para o usuário?
            # gr.Warning("Arquivo de vídeo não encontrado para download.") # Exemplo
            return None # Retorna None para gr.File indicar falha

    def on_stop_processing(self):
        global should_continue
        print("Recebido comando para interromper processamento.")
        should_continue = False # Sinaliza para os loops pararem

        # Tenta chamar a função stop específica do modelo, se existir
        # Precisaria saber qual modelo está ativo (talvez lendo self.model_dropdown.value?)
        # Isso é complexo porque o handler do botão stop não recebe o dropdown como input direto.
        # Uma abordagem seria armazenar o modelo ativo em um gr.State quando o processamento inicia.
        # Por ora, apenas sinaliza a variável global e retorna uma mensagem.

        # Zera a variável global para permitir novo processamento depois
        # (Depende da lógica: quer permitir restart imediato ou requer novo carregamento?)
        # Se quiser permitir novo start:
        # should_continue = True # Resetaria aqui? Ou no início do on_start_processing?
        # Melhor resetar no on_start_processing.

        return gr.update(value="Processamento interrompido pelo usuário.", visible=True)


    # ========================================
    # Implementações específicas dos modelos
    # ========================================

    def contador_stop_processing(self):
        # Apenas controla a flag global, que já é feito em on_stop_processing
        # Esta função específica pode não ser necessária se on_stop_processing for genérica
        print("ContadorDePessoas: stop chamado (ação via flag global).")
        pass # A flag global já foi setada

    def contador_start_processing(self, input_data, output_path, detect_class, model_file):
        # Função de processamento para o contador (como estava antes, com correções)
        global should_continue
        should_continue = True # Garante que o processamento pode começar

        if not isinstance(input_data, str) or not os.path.exists(input_data):
            print(f"Erro (Contador): input_data não é um caminho de vídeo válido: {input_data}")
            return None, None

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            print(f"Erro (Contador): Não foi possível abrir o vídeo: {input_data}")
            return None, None

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w == 0 or h == 0:
             print(f"Erro (Contador): Dimensões do vídeo inválidas (w={w}, h={h})")
             cap.release()
             return None, None

        frameArea = h * w
        areaTH = frameArea * 0.003

        leftmostLine = int(1.0 / 6 * w)
        rightmostLine = int(5.0 / 6 * w)
        leftmostLimit = int(1.0 / 12 * w)
        rightmostLimit = int(11.0 / 12 * w)
        leftmostLineColor = (255, 0, 0)
        rightmostLineColor = (0, 0, 255)

        pt1 = [rightmostLine, 0]; pt2 = [rightmostLine, int(h)]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pt3 = [leftmostLine, 0]; pt4 = [leftmostLine, int(h)]
        pts_L2 = np.array([pt3, pt4], np.int32)
        pt5 = [leftmostLimit, 0]; pt6 = [leftmostLimit, int(h)]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pt7 = [rightmostLimit, 0]; pt8 = [rightmostLimit, int(h)]
        pts_L4 = np.array([pt7, pt8], np.int32)

        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True) # Exemplo: habilitar sombras
        kernelOp = np.ones((3, 3), np.uint8)
        kernelCl = np.ones((9, 9), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        persons = []
        max_p_age = 5
        pid = 1
        leftCounter = 0
        rightCounter = 0

        output_video_filename = f"output_contador_{os.path.basename(input_data)}"
        output_video_path = os.path.join(output_path, output_video_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30 # Default FPS
        size = (int(w), int(h))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # ou 'avc1', 'H264'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, size, True)

        if not out.isOpened():
             print(f"Erro (Contador): Não foi possível inicializar VideoWriter para {output_video_path}")
             cap.release()
             return None, None

        print(f"Contador: Iniciando processamento do vídeo {input_data} para {output_video_path}")
        frame_count = 0
        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                print(f"Contador: Fim do vídeo ou erro de leitura no frame {frame_count}.")
                break

            frame_count += 1
            # Atualiza idade das pessoas rastreadas
            for per in persons:
                per.age_one()

            # Processamento de background subtraction
            fgmask = backgroundSubtractor.apply(frame)
            ret2, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_pids = [] # Para rastrear PIDs atualizados neste frame
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > areaTH:
                    M = cv2.moments(cnt)
                    # Evita divisão por zero se m00 for 0
                    if M['m00'] == 0: continue
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w_box, h_box = cv2.boundingRect(cnt)

                    newPerson = True
                    # Verifica se está dentro dos limites verticais onde o rastreamento ocorre
                    # (A lógica original parece usar cy, mas os limites são horizontais - verificar)
                    # Assumindo que os limites são verticais (eixo Y):
                    # if leftmostLimit < cx < rightmostLimit: # Se limites fossem horizontais
                    # Usando a lógica original que parece comparar cy com limites horizontais (posições X):
                    # Isso parece estranho. Talvez a intenção fosse limites verticais?
                    # Vamos manter a lógica original por enquanto:
                    if cy in range(int(h)): # Verifica se está dentro da altura do frame (redundante?)
                        for person in persons:
                             # Verifica se o bounding box atual se sobrepõe a uma pessoa existente
                             # Adiciona uma tolerância para facilitar a associação
                             tolerance = w_box * 0.5
                             if abs(cx - person.getX()) <= (w_box / 2 + tolerance) and \
                                abs(cy - person.getY()) <= (h_box / 2 + tolerance):

                                newPerson = False
                                person.updateCoords(cx, cy)
                                current_pids.append(person.getId()) # Marca como atualizado

                                # Verifica cruzamento de linha *apenas se ainda não foi contado*
                                if person.getState() == '0': # Se ainda não cruzou
                                    if person.goingLeft(rightmostLine, leftmostLine):
                                        leftCounter += 1
                                        print(f"ID {person.getId()} cruzou para Esquerda. Total Esq: {leftCounter}")
                                    elif person.goingRight(rightmostLine, leftmostLine):
                                        rightCounter += 1
                                        print(f"ID {person.getId()} cruzou para Direita. Total Dir: {rightCounter}")
                                break # Sai do loop interno, associou a uma pessoa

                        if newPerson:
                            # Cria nova pessoa apenas se não foi associada
                            person = Person(pid, cx, cy, max_p_age)
                            persons.append(person)
                            current_pids.append(pid)
                            print(f"Nova Pessoa detectada: ID {pid} em ({cx}, {cy})")
                            pid += 1

                    # Desenha bounding box e círculo para todas as detecções válidas
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1) # Verde para detecção
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1) # Vermelho no centroide

            # Limpa pessoas que não foram atualizadas (timedOut) ou saíram dos limites
            persons_to_remove = []
            for person in persons:
                 # Marca como 'Done' se saiu dos limites (verificar lógica dos limites novamente)
                 # A lógica original parece misturar coordenadas X e Y nos limites. Revendo:
                 # Linhas são verticais (X fixo). Limites também parecem ser X.
                 # O movimento é horizontal (esquerda/direita), então a coordenada X é relevante.
                person_x = person.getX()
                if person.getState() == '1': # Se já cruzou uma linha
                     if person.getDir() == 'right' and person_x > rightmostLimit:
                          person.setDone()
                     elif person.getDir() == 'left' and person_x < leftmostLimit:
                          person.setDone()

                # Remove se timedOut ou se saiu da área de interesse e já foi contado
                # Adiciona verificação para não remover quem acabou de ser atualizado
                # if (person.timedOut() or person.isDone()) and person.getId() not in current_pids:
                if person.timedOut() and person.getId() not in current_pids: # Simplificado: remove apenas por timeout
                     persons_to_remove.append(person)
                     print(f"Removendo Pessoa ID {person.getId()} por timeout.")

            # Remove as pessoas marcadas fora do loop principal de iteração
            for p_rem in persons_to_remove:
                persons.remove(p_rem)
                del p_rem

            # Desenha informações e linhas no frame
            leftMsg = f'Esquerda: {leftCounter}'
            rightMsg = f'Direita: {rightCounter}'
            cv2.polylines(frame, [pts_L1], False, rightmostLineColor, thickness=1)
            cv2.polylines(frame, [pts_L2], False, leftmostLineColor, thickness=1)
            cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1) # Limites em branco
            cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
            cv2.putText(frame, leftMsg, (10, 20), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA) # Contorno preto
            cv2.putText(frame, leftMsg, (10, 20), font, 0.6, leftmostLineColor, 1, cv2.LINE_AA) # Texto azul
            cv2.putText(frame, rightMsg, (10, 45), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA) # Contorno preto
            cv2.putText(frame, rightMsg, (10, 45), font, 0.6, rightmostLineColor, 1, cv2.LINE_AA) # Texto vermelho

            out.write(frame) # Escreve o frame

            # Adiciona um pequeno sleep para não sobrecarregar CPU (opcional)
            # time.sleep(0.01)

        # Fim do loop while
        print(f"Contador: Processamento concluído para {input_data}. Vídeo salvo em {output_video_path}")
        cap.release()
        out.release()
        cv2.destroyAllWindows() # Garante que janelas (se houver) são fechadas

        # Retorna o caminho do vídeo e um placeholder (a função on_start_processing espera 2 valores)
        return output_video_path, output_video_path


    def contador_process_webcam_frame(self, frame, detect_class, model_file):
        # Implementação para webcam seria muito diferente, pois não tem começo/fim definido
        # e o background subtractor precisa se adaptar continuamente.
        # Retornando o frame original por enquanto.
        print("Função contador_process_webcam_frame não implementada.")
        return frame

    # Funções do FaceMash (Comentadas ou removidas se não forem usadas)
    # ...


# Bloco principal para iniciar a aplicação
if __name__ == "__main__":
    # Garante que a variável global existe antes de ser usada
    should_continue = True
    # Instancia a classe da aplicação
    app_instance = App()
    # Inicia a interface Gradio
    print("Iniciando a interface Gradio...")
    app_instance.demo.launch(server_name="0.0.0.0", server_port=7860, share=False) # share=True pode causar problemas às vezes
    print("Interface Gradio encerrada.")