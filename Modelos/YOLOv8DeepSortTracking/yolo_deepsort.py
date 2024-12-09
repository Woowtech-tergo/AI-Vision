# yolo_deepsort.py

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import gradio as gr
import deep_sort.deep_sort.deep_sort as ds

# Controla se o processamento deve ser interrompido
should_continue = True

# Variáveis globais para modelo, tracker e contador de IDs únicos
model = None
tracker = None
unique_ids = set()

def get_detectable_classes(model_file):
    """Obtém as classes detectáveis de um arquivo de modelo fornecido.

    Parâmetros:
    - model_file: Nome do arquivo do modelo.

    Retorna:
    - class_names: Lista dos nomes das classes detectáveis.
    """
    global model
    if model is None:
        model = YOLO(model_file)
    class_names = list(model.names.values())  # Obtém diretamente a lista de nomes das classes
    return class_names

# Função para interromper o processamento de vídeo
def stop_processing():
    global should_continue
    should_continue = False  # Altera a variável global para parar o processamento
    return "Processamento interrompido."

# Função para iniciar o processamento de vídeo
def start_processing(input_data, output_path, detect_class, model_file, progress=gr.Progress(track_tqdm=True)):
    global should_continue, model, tracker, unique_ids
    should_continue = True

    # Reinicia o conjunto de IDs únicos
    unique_ids = set()

    # Obter a lista de classes detectáveis
    detect_classes = get_detectable_classes(model_file)
    # Encontrar o índice da classe selecionada
    detect_class_index = detect_classes.index(detect_class)

    if model is None:
        model = YOLO(model_file)
    if tracker is None:
        tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    if isinstance(input_data, str):
        # Input é um arquivo de vídeo
        output_video_path = detect_and_track(input_data, output_path, detect_class_index, model, tracker)
        return output_video_path, output_video_path
    else:
        # Input é um frame da webcam
        # O processamento é feito em tempo real na função process_webcam_frame
        return None, None

def putTextWithBackground(img, text, origin, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, text_color=(255, 255, 255),
                          bg_color=(0, 0, 0), thickness=1):
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

def extract_detections(results, detect_class_index):
    """
    Extrai e processa informações de detecção dos resultados do modelo.
    - results: Resultados da previsão do modelo YOLOv8.
    - detect_class_index: Índice da classe alvo a ser extraída.
    """
    detections = np.empty((0, 4))
    confarray = []

    for r in results:
        for box in r.boxes:
            if box.cls[0].int() == detect_class_index:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()
                conf = round(box.conf[0].item(), 2)
                detections = np.vstack((detections, np.array([x1, y1, x2, y2])))
                confarray.append(conf)
    return detections, confarray

# Processamento de vídeo
def detect_and_track(input_path: str, output_path: str, detect_class_index: int, model, tracker) -> Path:
    """
    Processa o vídeo, detecta e rastreia os objetos.
    - input_path: Caminho para o arquivo de vídeo de entrada.
    - output_path: Caminho para salvar o vídeo processado.
    - detect_class_index: Índice da classe alvo a ser detectada e rastreada.
    - model: Modelo utilizado para detecção de objetos.
    - tracker: Modelo utilizado para rastreamento de objetos.
    """
    global should_continue, unique_ids  # Garante que estamos modificando a variável global

    cap = cv2.VideoCapture(input_path)  # Abre o arquivo de vídeo com OpenCV.
    if not cap.isOpened():  # Verifica se o vídeo foi aberto com sucesso.
        print(f"Erro ao abrir o arquivo de vídeo {input_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Obtém o número total de quadros do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtém a taxa de quadros do vídeo
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )  # Obtém a resolução do vídeo (largura e altura).
    output_video_path = Path(output_path) / "output.mp4"  # Define o caminho de saída para o vídeo processado.

    # Configura o formato de codificação do vídeo como MP4 no formato 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(
        str(output_video_path), fourcc, fps, size, isColor=True
    )  # Cria um objeto VideoWriter para salvar o vídeo.

    frame_count = 0  # Contador de frames processados

    # Lê e processa cada quadro do vídeo
    # Mostra o progresso utilizando tqdm
    for _ in tqdm(range(total_frames)):
        # Interrompe o processamento caso o botão "Interromper Processamento" seja pressionado na GUI
        if not should_continue:
            print("Interrompendo o processamento")
            break

        success, frame = cap.read()  # Lê quadro por quadro do vídeo

        # Se a leitura falhar (fim do vídeo ou erro), interrompe o loop
        if not success:
            print("Fim do vídeo ou erro ao ler o frame")
            break

        # Realiza a detecção de objetos no quadro atual utilizando o modelo YOLOv8.
        results = model(frame)

        # Extrai informações de detecção dos resultados.
        detections, confarray = extract_detections(results, detect_class_index)

        # Realiza o rastreamento dos objetos detectados com o modelo DeepSort.
        resultsTracker = tracker.update(detections, confarray, frame)

        # Atualiza o conjunto de IDs únicos
        for x1, y1, x2, y2, Id in resultsTracker:
            unique_ids.add(Id)

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Converte as posições para inteiros.

            # Desenha bounding boxes e texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(
                frame,
                str(int(Id)),
                (max(-10, x1), max(40, y1)),
                font_scale=1.5,
                text_color=(255, 255, 255),
                bg_color=(255, 0, 255),
            )

        # Exibe o contador de objetos únicos no frame
        total_count = len(unique_ids)
        putTextWithBackground(
            frame,
            f"Total de objetos: {total_count}",
            (10, 30),
            font_scale=1.0,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )

        output_video.write(frame)  # Escreve o quadro processado no arquivo de saída.

        frame_count += 1  # Incrementa o contador de frames processados

    output_video.release()  # Libera o objeto VideoWriter.
    cap.release()  # Libera o arquivo de vídeo.

    print(f"Processamento concluído. Total de frames processados: {frame_count}")
    print(f"O diretório de saída é: {output_video_path}")
    print(f"Total de objetos únicos detectados: {len(unique_ids)}")
    return output_video_path

# Função para processar frames da webcam
def process_webcam_frame(image, detect_class, model_file):
    """Processa um frame da webcam."""
    global model, tracker, unique_ids

    # Obter a lista de classes detectáveis
    detect_classes = get_detectable_classes(model_file)
    # Encontrar o índice da classe selecionada
    detect_class_index = detect_classes.index(detect_class)

    if model is None:
        model = YOLO(model_file)
    if tracker is None:
        tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    if 'unique_ids' not in globals():
        unique_ids = set()

    # Converte a imagem para o formato correto
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Realiza a detecção de objetos no quadro atual utilizando o modelo YOLOv8.
    results = model(frame)

    # Extrai informações de detecção dos resultados.
    detections, confarray = extract_detections(results, detect_class_index)

    # Realiza o rastreamento dos objetos detectados com o modelo DeepSort.
    resultsTracker = tracker.update(detections, confarray, frame)

    # Atualiza o conjunto de IDs únicos
    for x1, y1, x2, y2, Id in resultsTracker:
        unique_ids.add(Id)

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Converte as posições para inteiros.

        # Desenha bounding boxes e texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        putTextWithBackground(
            frame,
            str(int(Id)),
            (max(-10, x1), max(40, y1)),
            font_scale=1.5,
            text_color=(255, 255, 255),
            bg_color=(255, 0, 255),
        )

    # Exibe o contador de objetos únicos no frame
    total_count = len(unique_ids)
    putTextWithBackground(
        frame,
        f"Total de Pessoas no Video: {total_count}",
        (10, 30),
        font_scale=1.0,
        text_color=(255, 255, 255),
        bg_color=(0, 0, 0),
    )

    # Converte a imagem de volta para RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame
