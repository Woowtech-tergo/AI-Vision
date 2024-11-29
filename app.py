from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm

import deep_sort.deep_sort.deep_sort as ds

import gradio as gr

# Controla se o processamento deve ser interrompido
should_continue = True


def get_detectable_classes(model_file):
    """Obtém as classes detectáveis de um arquivo de modelo fornecido.

    Parâmetros:
    - model_file: Nome do arquivo do modelo.

    Retorna:
    - class_names: Lista dos nomes das classes detectáveis.
    """
    model = YOLO(model_file)
    class_names = list(model.names.values())  # Obtém diretamente a lista de nomes das classes
    del model  # Exclui a instância do modelo para liberar recursos
    return class_names


# Função para interromper o processamento de vídeo
def stop_processing():
    global should_continue
    should_continue = False  # Altera a variável global para parar o processamento
    return "Tentando interromper o processamento..."


# Função para iniciar o processamento de vídeo
# gr.Progress(track_tqdm=True) captura a barra de progresso tqdm para exibição na interface GUI
def start_processing(input_path, output_path, detect_class, model, progress=gr.Progress(track_tqdm=True)):
    global should_continue
    should_continue = True

    detect_class = int(detect_class)
    model = YOLO(model)
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    output_video_path = detect_and_track(input_path, output_path, detect_class, model, tracker)
    return output_video_path, output_video_path


def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=1):
    """Desenha texto com fundo.

    :param img: Imagem de entrada.
    :param text: Texto a ser desenhado.
    :param origin: Coordenadas do canto superior esquerdo do texto.
    :param font: Tipo de fonte.
    :param font_scale: Tamanho da fonte.
    :param text_color: Cor do texto.
    :param bg_color: Cor do fundo.
    :param thickness: Espessura da linha do texto.
    """
    # Calcula o tamanho do texto
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Desenha o retângulo de fundo
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # Subtrai 5 para adicionar margem
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # Desenha o texto sobre o retângulo
    text_origin = (origin[0], origin[1] - 5)  # Ajusta a posição para adicionar margem
    cv2.putText(
        img,
        text,
        text_origin,
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def extract_detections(results, detect_class):
    """
    Extrai e processa informações de detecção dos resultados do modelo.
    - results: Resultados da previsão do modelo YoloV8, incluindo posição, categoria e confiança dos objetos detectados.
    - detect_class: Índice da classe alvo a ser extraída.
    Referência: https://docs.ultralytics.com/modes/predict/#working-with-results
    """
    # Inicializa um array numpy bidimensional vazio para armazenar as posições dos objetos detectados
    detections = np.empty((0, 4))

    confarray = []  # Inicializa uma lista vazia para armazenar os valores de confiança dos objetos detectados.

    # Itera sobre os resultados das detecções
    for r in results:
        for box in r.boxes:
            # Verifica se a classe detectada corresponde à classe desejada e extrai informações
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()  # Extrai as posições e converte para lista de inteiros.
                conf = round(box.conf[0].item(), 2)  # Extrai a confiança e arredonda para duas casas decimais.
                detections = np.vstack(
                    (detections, np.array([x1, y1, x2, y2])))  # Adiciona as posições ao array de detecções.
                confarray.append(conf)  # Adiciona a confiança à lista.
    return detections, confarray  # Retorna as posições e as confianças.


# Processamento de vídeo
def detect_and_track(input_path: str, output_path: str, detect_class: int, model, tracker) -> Path:
    """
    Processa o vídeo, detecta e rastreia os objetos.
    - input_path: Caminho para o arquivo de vídeo de entrada.
    - output_path: Caminho para salvar o vídeo processado.
    - detect_class: Índice da classe alvo a ser detectada e rastreada.
    - model: Modelo utilizado para detecção de objetos.
    - tracker: Modelo utilizado para rastreamento de objetos.
    """
    global should_continue
    cap = cv2.VideoCapture(input_path)  # Abre o arquivo de vídeo com OpenCV.
    if not cap.isOpened():  # Verifica se o vídeo foi aberto com sucesso.
        print(f"Erro ao abrir o arquivo de vídeo {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtém o número total de quadros do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtém a taxa de quadros do vídeo
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Obtém a resolução do vídeo (largura e altura).
    output_video_path = Path(output_path) / "output.avi"  # Define o caminho de saída para o vídeo processado.

    # Configura o formato de codificação do vídeo como AVI no formato XVID
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size,
                                   isColor=True)  # Cria um objeto VideoWriter para salvar o vídeo.

    # Lê e processa cada quadro do vídeo
    # Mostra o progresso utilizando tqdm
    for _ in tqdm(range(total_frames)):
        # Interrompe o processamento caso o botão "Stop" seja pressionado na GUI
        if not should_continue:
            print('Interrompendo o processamento')
            break

        success, frame = cap.read()  # Lê quadro por quadro do vídeo

        # Se a leitura falhar (fim do vídeo ou erro), interrompe o loop
        if not success:
            break

        # Realiza a detecção de objetos no quadro atual utilizando o modelo YoloV8.
        results = model(frame, stream=True)

        # Extrai informações de detecção dos resultados.
        detections, confarray = extract_detections(results, detect_class)

        # Realiza o rastreamento dos objetos detectados com o modelo DeepSort.
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Converte as posições para inteiros.

            # Desenha bounding boxes e texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5,
                                  text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)  # Escreve o quadro processado no arquivo de saída.

    output_video.release()  # Libera o objeto VideoWriter.
    cap.release()  # Libera o arquivo de vídeo.

    print(f'O diretório de saída é: {output_video_path}')
    return output_video_path


if __name__ == "__main__":
    # Lista oficial de modelos YoloV8 e V9. O modelo será baixado automaticamente na primeira execução.
    # model_list = ["yolov9c.pt", "yolov9e", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    model_list = ["yolov9c.pt","yolov8n.pt"]

    # Obtém todas as classes detectáveis pelo modelo YoloV8. Por padrão, usa o primeiro modelo da lista.
    detect_classes = get_detectable_classes(model_list[0])

    # Exemplo de entrada para a interface Gradio, incluindo um caminho de vídeo de teste, um diretório de saída temporário, uma classe de detecção e um modelo.
    examples = [["test.mp4", tempfile.mkdtemp(), detect_classes[0], model_list[0]]]

    # Cria uma interface GUI utilizando Gradio
    with gr.Blocks() as demo:
        with gr.Tab("Tracking"):
            gr.Markdown(
                """
                # Detecção e Rastreamento de Objetos
                Baseado em OpenCV + YoloV8 + DeepSort
                """
            )
            with gr.Row():
                with gr.Column():
                    input_path = gr.Video(label="Vídeo de Entrada")  # Controle para upload de vídeo

                    model = gr.Dropdown(model_list,
                                        value=model_list[0],
                                        label="Modelo")  # Use o primeiro modelo por padrão

                    detect_class = gr.Dropdown(detect_classes,
                                               value=detect_classes[0],
                                               label="Classe",
                                               type='index')  # Use a primeira classe por padrão

                    # model = gr.Dropdown(model_list,
                    #                     value=0,
                    #                     label="Modelo")  # Menu suspenso para seleção de modelo
                    # detect_class = gr.Dropdown(detect_classes,
                    #                            value=0,
                    #                            label="Classe",
                    #                            type='index')  # Menu suspenso para seleção de classe alvo
                    output_dir = gr.Textbox(label="Diretório de Saída",
                                            value=tempfile.mkdtemp())  # Caixa de texto para especificar o diretório de saída
                    with gr.Row():
                        start_button = gr.Button("Iniciar Processamento")  # Botão para iniciar o processamento
                        stop_button = gr.Button("Interromper Processamento")  # Botão para interromper o processamento
                with gr.Column():
                    output = gr.Video()  # Controle para exibir o vídeo processado
                    output_path = gr.Textbox(
                        label="Caminho do Arquivo de Saída")  # Caixa de texto para exibir o caminho do arquivo de saída
                    gr.Examples(
                        examples,
                        label="Exemplos",
                        inputs=[input_path, output_dir, detect_class, model],
                        outputs=[output, output_path],
                        fn=start_processing,
                        cache_examples=False,
                    )
        # Conecta os botões às funções de processamento
        start_button.click(start_processing, inputs=[input_path, output_dir, detect_class, model],
                           outputs=[output, output_path])
        stop_button.click(stop_processing)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
