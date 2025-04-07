import streamlit as st
import os
import time
import cv2
import numpy as np

# IMPORTS DOS MODELOS (ajuste conforme a sua estrutura de pacotes)
# Se a pasta Modelos tiver um __init__.py e um YOLOv8DeepSortTracking.py, etc., isso deve funcionar:
try:
    from ultralytics import YOLO
    import deep_sort.deep_sort.deep_sort as ds
    from Modelos.YOLOv8DeepSortTracking import yolo_deepsort
    from Modelos.ContadorDePessoasEmVideo.Person import Person
    ULTRA_OK = True
except ImportError as e:
    st.warning(f"Não foi possível importar ultralytics/DeepSORT ou arquivos em 'Modelos': {e}")
    ULTRA_OK = False


# =====================================
# Flag Global para interromper loops
# =====================================
should_continue = True


# =====================================
# Funções 'start_processing' do YOLOv8DeepSort
# e do ContadorDePessoasEmVideo
# (caso você tenha no outro arquivo, pode chamar aqui diretamente)
# =====================================

def yolo_start_processing(input_data, output_dir, detect_class, model_file):
    """
    Invoca a função do yolo_deepsort, se disponível.
    """
    global should_continue
    should_continue = True  # Reseta a flag toda vez que inicia

    if not ULTRA_OK:
        st.error("YOLO/DeepSort não disponíveis. Verifique imports.")
        return None, None

    # Chama a função do seu módulo YOLOv8DeepSortTracking
    return yolo_deepsort.start_processing(input_data, output_dir, detect_class, model_file)


def yolo_stop_processing():
    """
    Interrompe o loop de YOLO + DeepSort, se houver essa lógica no yolo_deepsort.
    """
    if not ULTRA_OK:
        return
    yolo_deepsort.stop_processing()


def contador_start_processing(input_data, output_dir, detect_class, model_file):
    """
    Mesma função que estava dentro da classe App no Gradio
    Adaptada para Streamlit.
    """
    global should_continue
    should_continue = True

    # Verifica se input_data é um arquivo de vídeo válido
    if not isinstance(input_data, str) or not os.path.exists(input_data):
        st.error(f"Não é um caminho de vídeo válido: {input_data}")
        return None, None

    cap = cv2.VideoCapture(input_data)
    if not cap.isOpened():
        st.error(f"Não foi possível abrir o vídeo: {input_data}")
        return None, None

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameArea = w * h
    areaTH = frameArea * 0.003

    leftmostLine = int(1.0 / 6 * w)
    rightmostLine = int(5.0 / 6 * w)
    leftmostLimit = int(1.0 / 12 * w)
    rightmostLimit = int(11.0 / 12 * w)

    leftmostLineColor = (255, 0, 0)
    rightmostLineColor = (0, 0, 255)

    pt1 = [rightmostLine, 0]
    pt2 = [rightmostLine, int(h)]
    pts_L1 = np.array([pt1, pt2], np.int32)
    pt3 = [leftmostLine, 0]
    pt4 = [leftmostLine, int(h)]
    pts_L2 = np.array([pt3, pt4], np.int32)

    pt5 = [leftmostLimit, 0]
    pt6 = [leftmostLimit, int(h)]
    pts_L3 = np.array([pt5, pt6], np.int32)
    pt7 = [rightmostLimit, 0]
    pt8 = [rightmostLimit, int(h)]
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

    # Preparar o arquivo de saída
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "contador_output.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    size = (int(w), int(h))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size, True)

    while cap.isOpened() and should_continue:
        ret, frame = cap.read()
        if not ret:
            break

        # Envelhece todos os persons
        for per in persons:
            per.age_one()

        fgmask = backgroundSubtractor.apply(frame)
        _, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > areaTH:
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                x, y, w_box, h_box = cv2.boundingRect(cnt)

                newPerson = True
                # Check se está nos limites horizontais
                if leftmostLimit < cx < rightmostLimit:
                    for person in persons:
                        # Checa proximidade pra ver se é o mesmo person
                        if abs(cx - person.getX()) <= w_box and abs(cy - person.getY()) <= h_box:
                            newPerson = False
                            person.updateCoords(cx, cy)
                            # Checa se cruzou linhas
                            if person.goingLeft(rightmostLine, leftmostLine):
                                leftCounter += 1
                            elif person.goingRight(rightmostLine, leftmostLine):
                                rightCounter += 1
                            break

                    if newPerson:
                        # Cria um novo Person
                        p = Person(pid, cx, cy, max_p_age)
                        persons.append(p)
                        pid += 1

                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Desenhar linhas
        cv2.polylines(frame, [pts_L1], False, rightmostLineColor, thickness=2)
        cv2.polylines(frame, [pts_L2], False, leftmostLineColor, thickness=2)
        cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
        cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)

        leftMsg = f"Esquerda: {leftCounter}"
        rightMsg = f"Direita: {rightCounter}"
        cv2.putText(frame, leftMsg, (10, 40), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, leftMsg, (10, 40), font, 0.6, leftmostLineColor, 1, cv2.LINE_AA)

        cv2.putText(frame, rightMsg, (10, 80), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, rightMsg, (10, 80), font, 0.6, rightmostLineColor, 1, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()

    return output_video_path, output_video_path


def contador_stop_processing():
    global should_continue
    should_continue = False


# =====================================
# Dicionário de Funções para cada modelo
# =====================================

model_functions = {
    "YOLOv8DeepSort": {
        "start_processing": yolo_start_processing,
        "stop_processing": yolo_stop_processing,
        "model_file": "yolov8n.pt",  # Ajuste se necessário
        "get_detectable_classes": lambda _: ["person", "car", "dog"]  # Exemplo
    },
    "ContadorDePessoasEmVideo": {
        "start_processing": contador_start_processing,
        "stop_processing": contador_stop_processing,
        "model_file": None,
        "get_detectable_classes": lambda _: ["person"]
    }
}


def get_detect_classes(model_name):
    info = model_functions.get(model_name)
    if info and "get_detectable_classes" in info:
        return info["get_detectable_classes"](info["model_file"])
    return []


# =====================================
# Funções para Streamlit UI
# =====================================

def process_file_with_model(input_video_path, model_name, selected_class):
    """
    Função principal de processamento invocada ao clicar no botão "Processar".
    - input_video_path: caminho local do vídeo salvo
    - model_name: qual modelo?
    - selected_class: classe escolhida no dropdown (se aplicável)
    """
    start_time = time.time()

    model_info = model_functions.get(model_name)
    if not model_info:
        st.error("Modelo não encontrado.")
        return None, None, "Erro: modelo não encontrado."

    start_func = model_info["start_processing"]
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path, _ = start_func(input_video_path, output_dir, selected_class, model_info["model_file"])
    if output_path is None or not os.path.exists(output_path):
        st.error("Erro: O processamento não retornou um arquivo de vídeo válido.")
        return None, None, "Erro no processamento."

    elapsed = time.time() - start_time
    h, r = divmod(elapsed, 3600)
    m, s = divmod(r, 60)
    time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    return output_path, time_str, "Processamento concluído!"


def main():
    st.title("Aplicação de Visão Computacional em Streamlit")
    st.write("Carregue um vídeo, selecione o modelo e clique em 'Processar'.")

    # Fonte de entrada (se quiser webcam, teria que usar st.camera_input etc.)
    input_source = st.radio("Selecione a fonte de entrada", ["Arquivo de Vídeo"])
    # Se houver a possibilidade de webcam, adaptamos. Aqui só vídeo.

    uploaded_file = None
    if input_source == "Arquivo de Vídeo":
        uploaded_file = st.file_uploader("Carregue um arquivo de vídeo", type=["mp4", "avi", "mov", "mkv"])

    # Selecionar modelo
    model_name = st.selectbox("Modelo", list(model_functions.keys()))
    # Atualiza dropdown de classes detectáveis
    classes = get_detect_classes(model_name)
    if classes:
        selected_class = st.selectbox("Classe para detecção (caso YOLO)", classes)
    else:
        selected_class = "person"

    # Botões
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        run_process = st.button("Processar")
    with col2:
        stop_process = st.button("Parar Processamento")
    # (Em Streamlit, esse "Parar" só seta a global, mas o loop em OpenCV é quem respeita isso.)

    # Exibir resultados
    if run_process:
        if not uploaded_file:
            st.error("Nenhum arquivo de vídeo foi carregado.")
        else:
            # Salvar o vídeo em disco temporariamente
            temp_input_path = os.path.join("temp_dir", "input_upload.mp4")
            os.makedirs("temp_dir", exist_ok=True)
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.read())

            # Chamar a lógica de processamento
            output_path, elapsed_time, message = process_file_with_model(
                temp_input_path, model_name, selected_class
            )

            if output_path is not None:
                st.success(message)
                st.write(f"Tempo de processamento: {elapsed_time}")
                # Exibe o vídeo processado
                st.video(output_path)

                # Botão de download
                with open(output_path, "rb") as f:
                    vid_bytes = f.read()
                st.download_button(
                    label="Baixar Vídeo Processado",
                    data=vid_bytes,
                    file_name=os.path.basename(output_path),
                    mime="video/mp4"
                )

    if stop_process:
        # Chama a função de stop do modelo escolhido
        model_info = model_functions.get(model_name)
        if model_info and "stop_processing" in model_info:
            model_info["stop_processing"]()
            st.warning("Processamento interrompido.")
        else:
            st.warning("Não foi possível interromper ou modelo inválido.")


if __name__ == "__main__":
    main()
