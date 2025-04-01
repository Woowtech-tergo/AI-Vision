import tempfile
from pathlib import Path
import numpy as np
import cv2 # opencv-python
from ultralytics import YOLO

import deep_sort.deep_sort.deep_sort as ds


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

    # Desenha um retângulo de fundo
    bottom_left = origin
    top_right = (origin[0] + text_width, origin[1] - text_height - 5)  # Subtrai 5 para adicionar alguma margem
    cv2.rectangle(img, bottom_left, top_right, bg_color, -1)

    # Desenha o texto sobre o retângulo
    text_origin = (origin[0], origin[1] - 5)  # Ajusta a posição para adicionar margem
    cv2.putText(img, text, text_origin, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def extract_detections(results, detect_class):
    """
    Extrai e processa informações de detecção dos resultados do modelo.
    - results: Resultados da previsão do modelo YoloV8, incluindo posição, categoria e confiança dos objetos detectados.
    - detect_class: Índice da categoria alvo a ser extraída.
    Referência: https://docs.ultralytics.com/modes/predict/#working-with-results
    """

    # Inicializa um array numpy bidimensional vazio para armazenar as posições dos objetos detectados
    # Caso não haja objetos detectados da categoria desejada, a inicialização evita erros no tracker
    detections = np.empty((0, 4))

    confarray = []  # Inicializa uma lista vazia para armazenar os valores de confiança dos objetos detectados.

    # Itera sobre os resultados das detecções
    # Referência: https://docs.ultralytics.com/modes/predict/#working-with-results
    for r in results:
        for box in r.boxes:
            # Verifica se a categoria detectada corresponde à categoria desejada e extrai informações
            if box.cls[0].int() == detect_class:
                x1, y1, x2, y2 = box.xywh[0].int().tolist()  # Extrai as posições e converte para lista de inteiros.
                conf = round(box.conf[0].item(),
                             2)  # Extrai a confiança, converte para float e arredonda para duas casas decimais.
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
    - detect_class: Índice da categoria alvo a ser detectada e rastreada.
    - model: Modelo utilizado para detecção de objetos.
    - tracker: Modelo utilizado para rastreamento de objetos.
    """
    cap = cv2.VideoCapture(input_path)  # Abre o arquivo de vídeo com OpenCV.
    if not cap.isOpened():  # Verifica se o vídeo foi aberto com sucesso.
        print(f"Erro ao abrir o arquivo de vídeo {input_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtém a taxa de quadros do vídeo.
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Obtém a resolução do vídeo (largura e altura).
    output_video_path = Path(output_path) / "output.avi"  # Define o caminho de saída para o vídeo processado.

    # Configura o formato de codificação do vídeo como um arquivo AVI no formato XVID
    # Caso deseje usar o codec h264 ou salvar em outro formato, pode ser necessário baixar o openh264-1.8.0
    # Link para download: https://github.com/cisco/openh264/releases/tag/v1.8.0
    # Após o download, coloque o arquivo DLL no diretório atual
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, size,
                                   isColor=True)  # Cria um objeto VideoWriter para salvar o vídeo.

    # Lê e processa cada frame do vídeo
    while True:
        success, frame = cap.read()  # Lê frame por frame do vídeo.

        # Interrompe o loop caso não consiga ler o frame (fim do vídeo ou erro).
        if not success:
            break

        # Realiza detecção de objetos no frame atual utilizando o modelo YoloV8.
        results = model(frame, stream=True)

        # Extrai informações de detecção dos resultados.
        detections, confarray = extract_detections(results, detect_class)

        # Realiza rastreamento dos objetos detectados com o modelo DeepSort.
        resultsTracker = tracker.update(detections, confarray, frame)

        for x1, y1, x2, y2, Id in resultsTracker:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Converte as posições para inteiros.

            # Desenha bounding boxes e texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            putTextWithBackground(frame, str(int(Id)), (max(-10, x1), max(40, y1)), font_scale=1.5,
                                  text_color=(255, 255, 255), bg_color=(255, 0, 255))

        output_video.write(frame)  # Escreve o frame processado no arquivo de saída.

    output_video.release()  # Libera o objeto VideoWriter.
    cap.release()  # Libera o arquivo de vídeo.

    print(f'O diretório de saída é: {output_video_path}')
    return output_video_path


if __name__ == "__main__":
    # Define o caminho do vídeo de entrada.
    ######
    input_path = "test.mp4"
    ######

    # Diretório de saída, por padrão é o diretório temporário do sistema.
    output_path = tempfile.mkdtemp()  # Cria um diretório temporário para salvar o vídeo processado.

    # Carrega os pesos do modelo YOLOv8
    model = YOLO("yolov8n.pt")

    # Define a categoria de objeto a ser detectada e rastreada
    # O primeiro índice do modelo oficial YOLOv8 é 'person'
    detect_class = 0
    print(f"Detectando {model.names[detect_class]}")  # model.names retorna todas as categorias suportadas pelo modelo.

    # Carrega o modelo DeepSort
    tracker = ds.DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")

    detect_and_track(input_path, output_path, detect_class, model, tracker)

