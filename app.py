# app.py (Streamlit Refined Version)

import streamlit as st
import os
import time
import cv2
import numpy as np
import tempfile
from pathlib import Path
import traceback

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="AI Vision",
    layout="wide",  # Usa a largura total da página
    initial_sidebar_state="collapsed" # Opcional: esconde sidebar se não usar
)

# --- Importações dos Modelos e Lógica de Processamento ---
# Mantenha seus imports aqui, ajustando os caminhos conforme sua estrutura
# Adicionada verificação para evitar crash se import falhar
try:
    from ultralytics import YOLO
    import deep_sort.deep_sort.deep_sort as ds # Import se for usado diretamente
    from Modelos.YOLOv8DeepSortTracking import yolo_deepsort
    from Modelos.ContadorDePessoasEmVideo.Person import Person
    MODELS_AVAILABLE = True
    print("Módulos de modelos carregados com sucesso.")
except ImportError as e:
    st.error(f"Falha ao importar módulos necessários: {e}. A aplicação pode não funcionar corretamente.")
    print(f"Erro Crítico nos Imports: {e}")
    MODELS_AVAILABLE = False
    # Definir mocks ou tratar erro pode ser necessário se quiser que a UI carregue mesmo assim
    # Exemplo:
    class MockYoloDeepSort:
        def start_processing(*args, **kwargs): raise NotImplementedError("Módulo YOLO/DeepSort não carregado")
        def stop_processing(*args, **kwargs): pass
    yolo_deepsort = MockYoloDeepSort()


# --- Flag Global para Interrupção ---
# Controla os loops de processamento de vídeo
should_continue = True


# --- Funções de Processamento (Adaptadas do seu código funcional) ---

def yolo_start_processing(input_data, output_dir, detect_class, model_file="yolov8n.pt"):
    """
    Invoca a função de processamento do YOLOv8 + DeepSort.

    Args:
        input_data (str): Caminho para o arquivo de vídeo de entrada.
        output_dir (str): Diretório onde o vídeo de saída será salvo.
        detect_class (str): A classe específica a ser detectada e rastreada.
        model_file (str): Caminho ou nome do arquivo do modelo YOLOv8.

    Returns:
        tuple: (caminho_do_video_de_saida, None) ou (None, None) em caso de erro.
               O segundo elemento é um placeholder para compatibilidade.
    """
    global should_continue
    should_continue = True  # Garante que um novo processamento possa começar

    if not MODELS_AVAILABLE or not hasattr(yolo_deepsort, 'start_processing'):
        st.error("Erro: Módulo YOLOv8/DeepSort não carregado ou função 'start_processing' não encontrada.")
        print("Erro: Tentativa de chamar yolo_start_processing sem o módulo/função disponível.")
        return None, None

    st.info(f"Iniciando YOLOv8 + DeepSort para classe '{detect_class}'...")
    print(f"Chamando yolo_deepsort.start_processing: input='{input_data}', output_dir='{output_dir}', class='{detect_class}', model='{model_file}'")
    try:
        # Chama a função do seu módulo, passando a flag global se ele a utilizar internamente
        # Se a sua função yolo_deepsort.start_processing não usar 'should_continue', remova o argumento
        # Adapte os argumentos conforme a assinatura exata da sua função
        # output_path, other_val = yolo_deepsort.start_processing(input_data, output_dir, detect_class, model_file, should_continue_flag=should_continue)

        # Assumindo que a função do módulo já usa a flag global ou tem sua própria lógica de parada
        output_path, other_val = yolo_deepsort.start_processing(input_data, output_dir, detect_class, model_file)

        print(f"yolo_deepsort.start_processing retornou: {output_path}")
        if output_path and os.path.exists(output_path):
            return output_path, other_val # Retorna o caminho e o segundo valor (se houver)
        else:
            st.error("Processamento YOLOv8 concluído, mas arquivo de saída não foi encontrado.")
            print(f"Erro: Arquivo de saída esperado não encontrado após processamento YOLOv8: {output_path}")
            return None, None
    except Exception as e:
        st.error(f"Erro durante a execução do YOLOv8 + DeepSort: {e}")
        print(f"Erro em yolo_start_processing: {e}")
        traceback.print_exc()
        return None, None


def yolo_stop_processing():
    """
    Chama a função de parada do módulo YOLOv8 + DeepSort, se existir.
    Alternativamente, apenas seta a flag global.
    """
    global should_continue
    should_continue = False
    print("Flag 'should_continue' setada para False (tentativa de parada).")
    if MODELS_AVAILABLE and hasattr(yolo_deepsort, 'stop_processing'):
        try:
            print("Chamando yolo_deepsort.stop_processing()")
            yolo_deepsort.stop_processing() # Chama a função específica do módulo, se houver
        except Exception as e:
            print(f"Erro ao chamar yolo_deepsort.stop_processing: {e}")


def contador_start_processing(input_data, output_dir, detect_class=None, model_file=None):
    """
    Executa a lógica de contagem de pessoas usando subtração de fundo.

    Args:
        input_data (str): Caminho para o vídeo de entrada.
        output_dir (str): Diretório para salvar o vídeo de saída.
        detect_class (str, optional): Ignorado nesta função. Defaults to None.
        model_file (str, optional): Ignorado nesta função. Defaults to None.

    Returns:
        tuple: (caminho_do_video_de_saida, caminho_do_video_de_saida) ou (None, None) em caso de erro.
               Retorna o caminho duas vezes para compatibilidade com a chamada original.
    """
    global should_continue
    should_continue = True

    if not MODELS_AVAILABLE: # Checa se a classe Person foi importada
        st.error("Classe 'Person' não disponível. Verifique a importação de Modelos.")
        print("Erro: Tentativa de chamar contador_start_processing sem a classe Person.")
        return None, None

    print(f"Iniciando Contador de Pessoas: input='{input_data}', output_dir='{output_dir}'")
    if not isinstance(input_data, str) or not os.path.exists(input_data):
        st.error(f"Caminho de vídeo inválido para o contador: {input_data}")
        print(f"Erro: Caminho inválido em contador_start_processing: {input_data}")
        return None, None

    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {input_data}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0: raise ValueError(f"Dimensões inválidas (w={w}, h={h})")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frameArea = w * h
        areaTH = frameArea * 0.003 # Threshold de área
        # Definições de linhas e limites (ajuste conforme necessidade)
        leftmostLine = int(1.0 / 6 * w); rightmostLine = int(5.0 / 6 * w)
        leftmostLimit = int(1.0 / 12 * w); rightmostLimit = int(11.0 / 12 * w)
        leftmostLineColor = (255, 0, 0); rightmostLineColor = (0, 0, 255) # Azul, Vermelho BGR

        # Arrays numpy para desenhar linhas
        pt1 = [rightmostLine, 0]; pt2 = [rightmostLine, h]; pts_L1 = np.array([pt1, pt2], np.int32)
        pt3 = [leftmostLine, 0]; pt4 = [leftmostLine, h]; pts_L2 = np.array([pt3, pt4], np.int32)
        pt5 = [leftmostLimit, 0]; pt6 = [leftmostLimit, h]; pts_L3 = np.array([pt5, pt6], np.int32)
        pt7 = [rightmostLimit, 0]; pt8 = [rightmostLimit, h]; pts_L4 = np.array([pt7, pt8], np.int32)

        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernelOp = np.ones((3, 3), np.uint8); kernelCl = np.ones((9, 9), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        persons = [] # Lista para armazenar objetos Person
        max_p_age = 5; pid = 1
        leftCounter = 0; rightCounter = 0

        # Configura arquivo de saída
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"contador_{Path(input_data).stem}.mp4"
        output_video_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not out.isOpened(): raise ValueError("Falha ao abrir VideoWriter")

        st_progress_bar = st.progress(0, text="Processando Contador...")
        frame_count = 0

        while cap.isOpened() and should_continue: # Verifica a flag global
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # --- Lógica de processamento do frame ---
            for per in persons: per.age_one()
            fgmask = backgroundSubtractor.apply(frame)
            _, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_pids = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > areaTH:
                    M = cv2.moments(cnt)
                    if M['m00'] == 0: continue
                    cx = int(M['m10'] / M['m00']); cy = int(M['m01'] / M['m00'])
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    newPerson = True
                    # A checagem de limites aqui é crucial e depende da sua intenção original
                    # Se for para rastrear apenas entre as linhas limites X:
                    if leftmostLimit < cx < rightmostLimit:
                        for person in persons:
                             # Lógica de associação (ajuste a tolerância se necessário)
                             tolerance = w_box * 0.3
                             if abs(cx - person.getX()) <= (w_box / 2 + tolerance) and \
                                abs(cy - person.getY()) <= (h_box / 2 + tolerance):
                                newPerson = False
                                person.updateCoords(cx, cy)
                                current_pids.append(person.getId())
                                # Lógica de contagem ao cruzar linhas (checar a classe Person)
                                if person.getState() == '0':
                                    if person.goingLeft(rightmostLine, leftmostLine):
                                        leftCounter += 1
                                        # person.setState('1') # Exemplo: marcar como contado
                                    elif person.goingRight(rightmostLine, leftmostLine):
                                        rightCounter += 1
                                        # person.setState('1')
                                break
                        if newPerson:
                            p = Person(pid, cx, cy, max_p_age) # Cria nova instância
                            persons.append(p)
                            current_pids.append(pid)
                            pid += 1
                    # Desenha independente da associação, se a área for suficiente
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # Limpeza
            persons_to_remove = []
            for person in persons:
                 person_x = person.getX()
                 # Marca como feito se saiu dos limites depois de cruzar
                 if person.getState() == '1':
                     if person.getDir() == 'right' and person_x > rightmostLimit: person.setDone()
                     elif person.getDir() == 'left' and person_x < leftmostLimit: person.setDone()
                 # Remove por timeout se não foi visto neste frame
                 if person.timedOut() and person.getId() not in current_pids:
                     persons_to_remove.append(person)
            for p_rem in persons_to_remove:
                try: persons.remove(p_rem)
                except ValueError: pass
                del p_rem

            # Desenhos finais
            leftMsg = f"Esq: {leftCounter}"; rightMsg = f"Dir: {rightCounter}"
            cv2.polylines(frame, [pts_L1], False, rightmostLineColor, thickness=1)
            cv2.polylines(frame, [pts_L2], False, leftmostLineColor, thickness=1)
            cv2.polylines(frame, [pts_L3], False, (255, 255, 255), thickness=1)
            cv2.polylines(frame, [pts_L4], False, (255, 255, 255), thickness=1)
            cv2.putText(frame, leftMsg, (10, 20), font, 0.6, (0,0,0), 3, cv2.LINE_AA); cv2.putText(frame, leftMsg, (10, 20), font, 0.6, leftmostLineColor, 1, cv2.LINE_AA)
            cv2.putText(frame, rightMsg, (10, 45), font, 0.6, (0,0,0), 3, cv2.LINE_AA); cv2.putText(frame, rightMsg, (10, 45), font, 0.6, rightmostLineColor, 1, cv2.LINE_AA)
            # --- Fim Lógica Frame ---

            out.write(frame)
            # Atualiza progresso
            if total_frames > 0:
                progress_percent = int(100 * frame_count / total_frames)
                st_progress_bar.progress(progress_percent, text=f"Processando Contador... {progress_percent}%")

        # Fim do loop
        st_progress_bar.progress(100, text="Processamento Contador Concluído!")
        print(f"Contador: Processamento finalizado. Output: {output_video_path}")
        return output_video_path, output_video_path # Retorna duas vezes por compatibilidade

    except Exception as e:
        st.error(f"Erro durante o processamento do Contador: {e}")
        print(f"Erro em contador_start_processing: {e}")
        traceback.print_exc()
        return None, None
    finally:
        # Garante que os recursos são liberados
        if cap and cap.isOpened():
            cap.release()
            print("Recurso VideoCapture liberado.")
        if out and out.isOpened():
            out.release()
            print("Recurso VideoWriter liberado.")
        cv2.destroyAllWindows()


def contador_stop_processing():
    """Seta a flag global para interromper o loop do contador."""
    global should_continue
    should_continue = False
    print("Flag 'should_continue' setada para False (tentativa de parada).")

# =====================================
# Dicionário de Funções (como no seu exemplo)
# =====================================
model_functions = {
    "YOLOv8DeepSort": {
        "start_processing": yolo_start_processing,
        "stop_processing": yolo_stop_processing,
        "model_file": "yolov8n.pt",
        # Função para obter classes (pode chamar o módulo original se disponível)
        "get_detectable_classes": lambda mf: yolo_deepsort.get_detectable_classes(mf) if MODELS_AVAILABLE and hasattr(yolo_deepsort, 'get_detectable_classes') else ["person", "car", "truck", "bus"]
    },
    "ContadorDePessoasEmVideo": {
        "start_processing": contador_start_processing,
        "stop_processing": contador_stop_processing,
        "model_file": None,
        "get_detectable_classes": lambda mf: ["person"] # Contador só detecta 'person' implicitamente
    }
}

def get_detect_classes(model_name):
    """Retorna a lista de classes detectáveis para o modelo selecionado."""
    info = model_functions.get(model_name)
    if info and "get_detectable_classes" in info:
        try:
            # Passa o model_file para a função lambda/original
            classes = info["get_detectable_classes"](info.get("model_file"))
            return classes if isinstance(classes, list) else []
        except Exception as e:
            print(f"Erro ao obter classes para {model_name}: {e}")
            return [] # Retorna vazio em caso de erro
    return []


# =====================================
# Função Principal da UI Streamlit
# =====================================
def main():
    """Função principal que monta a interface Streamlit."""
    st.title("👁️ AI Vision") # Título da página definido aqui

    # --- Inicialização do Estado da Sessão ---
    # Garante que as variáveis existem ao iniciar ou recarregar a página
    if 'model_name' not in st.session_state:
        st.session_state.model_name = list(model_functions.keys())[0]
    if 'selected_class' not in st.session_state:
        # Define uma classe inicial padrão (ex: 'person')
        initial_classes = get_detect_classes(st.session_state.model_name)
        st.session_state.selected_class = initial_classes[0] if initial_classes else None
    if 'uploaded_file_bytes' not in st.session_state:
        st.session_state.uploaded_file_bytes = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Aguardando vídeo..."
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False


    # --- Layout em Colunas ---
    col1, col2 = st.columns(2)

    # --- Coluna 1: Input e Controles ---
    with col1:
        st.header("Configuração")

        # Upload do Vídeo
        uploaded_file = st.file_uploader(
            "1. Carregue um arquivo de vídeo",
            type=["mp4", "avi", "mov", "mkv"],
            key="fileuploader",
            help="Formatos suportados: MP4, AVI, MOV, MKV"
            )

        # Atualiza o estado da sessão se um novo arquivo for carregado
        if uploaded_file is not None:
            # Compara pelo nome para evitar recarregar o mesmo arquivo em reruns
            if uploaded_file.name != st.session_state.get('uploaded_file_name'):
                st.session_state.uploaded_file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.output_path = None # Limpa resultado anterior
                st.session_state.processing_time = None
                st.session_state.status_message = "Vídeo carregado, pronto para processar."
                print(f"Arquivo '{uploaded_file.name}' carregado na memória.")

        # Exibe o vídeo carregado para visualização prévia, se existir
        if st.session_state.uploaded_file_bytes:
            st.write("Visualização do Vídeo Carregado:")
            st.video(st.session_state.uploaded_file_bytes)
        else:
             st.info("Nenhum vídeo carregado para visualização.")


        # Seleção de Modelo
        st.session_state.model_name = st.selectbox(
            "2. Selecione o Modelo:",
            list(model_functions.keys()),
            key='model_select',
            # index opcional para manter seleção anterior, mas pode causar erro se a lista mudar
            index=list(model_functions.keys()).index(st.session_state.model_name)
            )

        # Seleção de Classe (Dinâmico)
        available_classes = get_detect_classes(st.session_state.model_name)
        if available_classes:
             # Tenta manter a seleção anterior se ela ainda for válida, senão usa o primeiro
             current_selection_index = available_classes.index(st.session_state.selected_class) if st.session_state.selected_class in available_classes else 0
             st.session_state.selected_class = st.selectbox(
                f"3. Classe para Detecção ({st.session_state.model_name}):",
                available_classes,
                index=current_selection_index,
                key='class_select'
             )
        else:
            # Se não houver classes (ou modelo não requer), não mostra o selectbox
            st.caption(f"Nenhuma classe específica selecionável para {st.session_state.model_name}.")
            st.session_state.selected_class = None # Garante que não tem valor inválido


        # Botões de Ação
        st.write("4. Ações:")
        process_clicked = st.button(
            "Processar Vídeo",
            key="process_btn",
            disabled=(st.session_state.uploaded_file_bytes is None or st.session_state.is_processing),
            help="Inicia o processamento do vídeo com as opções selecionadas."
            )

        stop_clicked = st.button(
            "Parar Processamento",
            key="stop_btn",
            disabled=not st.session_state.is_processing, # Habilitado apenas durante processamento
            help="Tenta interromper o processo atual (pode levar alguns instantes)."
            )

    # --- Coluna 2: Output e Download ---
    with col2:
        st.header("Resultado")

        # Mostra status ou resultado
        if st.session_state.is_processing:
            st.info("Processamento em andamento... Aguarde.")
            # A barra de progresso será mostrada aqui pela função do contador
        elif st.session_state.output_path and os.path.exists(st.session_state.output_path):
             st.success(st.session_state.status_message)
             if st.session_state.processing_time:
                 st.info(f"Tempo de processamento: {st.session_state.processing_time}")

             # Exibe o vídeo processado
             st.write("Vídeo Processado:")
             try:
                 with open(st.session_state.output_path, "rb") as f_out:
                     output_video_bytes = f_out.read()
                 st.video(output_video_bytes) # st.video tem controle de fullscreen

                 # Botão de Download
                 st.download_button(
                    label="Baixar Vídeo Processado",
                    data=output_video_bytes,
                    file_name=os.path.basename(st.session_state.output_path),
                    mime="video/mp4",
                    key='download_btn'
                 )
             except FileNotFoundError:
                  st.error(f"Erro: Arquivo de saída não encontrado em {st.session_state.output_path}")
                  print(f"Erro Crítico: Tentativa de ler/baixar arquivo inexistente: {st.session_state.output_path}")
             except Exception as e:
                  st.error(f"Erro ao exibir ou preparar download: {e}")
                  print(f"Erro ao ler/exibir/baixar: {e}")
                  traceback.print_exc()
        else:
            # Exibe a última mensagem de status se não estiver processando e não houver output
            st.info(st.session_state.status_message)


    # --- Lógica de Controle ---
    # Processar
    if process_clicked and st.session_state.uploaded_file_bytes:
        st.session_state.is_processing = True
        st.session_state.output_path = None # Limpa resultado anterior
        st.session_state.processing_time = None
        st.session_state.status_message = "Iniciando processamento..."
        print("Botão 'Processar' clicado.")

        # Cria diretório temporário seguro
        temp_dir = tempfile.mkdtemp(prefix="st_vid_in_")
        temp_input_path = os.path.join(temp_dir, st.session_state.uploaded_file_name)
        try:
            print(f"Salvando vídeo carregado em: {temp_input_path}")
            with open(temp_input_path, "wb") as f:
                f.write(st.session_state.uploaded_file_bytes)

            # Chamar a função de processamento principal (adaptada do seu código)
            # Usa st.spinner para feedback visual durante a execução síncrona
            with st.spinner(f"Executando {st.session_state.model_name}..."):
                 output_path, elapsed_time, message = process_file_with_model(
                    temp_input_path,
                    st.session_state.model_name,
                    st.session_state.selected_class # Passa a classe selecionada
                 )

            # Atualiza o estado da sessão com os resultados
            st.session_state.output_path = output_path
            st.session_state.processing_time = elapsed_time
            st.session_state.status_message = message

        except Exception as e:
            st.session_state.status_message = f"Erro crítico durante setup/processamento: {e}"
            st.error(st.session_state.status_message)
            print(st.session_state.status_message)
            traceback.print_exc()
            st.session_state.output_path = None # Garante limpeza em caso de erro
        finally:
            # Limpa o arquivo temporário de input após o uso
            if os.path.exists(temp_input_path):
                 try:
                     os.remove(temp_input_path)
                     os.rmdir(temp_dir) # Remove o diretório temporário
                     print(f"Arquivo/Diretório temporário removido: {temp_input_path}")
                 except OSError as e:
                     print(f"Aviso: Não foi possível remover arquivo/diretório temporário {temp_input_path}: {e}")

            st.session_state.is_processing = False # Finaliza o estado de processamento
            st.rerun() # Atualiza a interface para mostrar resultados ou erros


    # Parar
    if stop_clicked:
        print("Botão 'Parar' clicado.")
        model_info = model_functions.get(st.session_state.model_name)
        if model_info and "stop_processing" in model_info:
            try:
                model_info["stop_processing"]() # Chama a função de parada específica
                st.session_state.status_message = "Tentativa de interrupção enviada."
                st.warning(st.session_state.status_message)
                print(st.session_state.status_message)
                # Nota: O loop de processamento precisa checar a flag 'should_continue'
                # A UI pode não atualizar imediatamente até o próximo rerun ou fim do processo.
                # Considerar desabilitar o botão processar aqui também se a parada for assíncrona.
                # st.session_state.is_processing = False # Considerar se a parada é imediata
                st.rerun() # Tenta atualizar a UI
            except Exception as e:
                 error_msg = f"Erro ao tentar parar: {e}"
                 st.error(error_msg)
                 print(error_msg)
        else:
            st.error("Função de parada não definida para este modelo.")
            print("Erro: Função stop_processing não encontrada.")


# =====================================
# Função Helper (adaptada do seu código)
# =====================================
def process_file_with_model(input_video_path, model_name, selected_class):
    """
    Orquestra o processamento do vídeo com o modelo selecionado.

    Args:
        input_video_path (str): Caminho para o vídeo de entrada (temporário).
        model_name (str): Nome do modelo selecionado.
        selected_class (str): Classe selecionada (relevante para YOLO).

    Returns:
        tuple: (caminho_saida, tempo_decorrido_str, mensagem_status)
               Retorna (None, None, mensagem_erro) em caso de falha.
    """
    start_time = time.time()
    output_path = None
    message = "Erro desconhecido."

    model_info = model_functions.get(model_name)
    if not model_info:
        message = "Erro: Modelo não encontrado no dicionário."
        print(message)
        return None, None, message

    start_func = model_info.get("start_processing")
    if not start_func:
         message = f"Erro: Função 'start_processing' não definida para {model_name}."
         print(message)
         return None, None, message

    output_dir = os.path.join(os.getcwd(), "outputs_st") # Diretório de saída
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Chama a função de processamento específica do modelo
        # Ela deve retornar (caminho_saida, outro_valor) ou (caminho_saida, caminho_saida)
        temp_output_path, _ = start_func(
            input_video_path,
            output_dir,
            selected_class, # Passa a classe selecionada
            model_info.get("model_file") # Passa o arquivo do modelo, se houver
            )

        # Verifica se o resultado é válido
        if temp_output_path and os.path.exists(temp_output_path):
            output_path = temp_output_path
            message = "Processamento concluído!"
            print(f"Processamento bem-sucedido, saída: {output_path}")
        else:
            message = "Erro: Processamento não gerou arquivo de saída válido."
            print(f"{message} Path retornado: {temp_output_path}")
            output_path = None

    except Exception as e:
        message = f"Erro durante processamento '{model_name}': {e}"
        print(message)
        traceback.print_exc()
        output_path = None # Garante que não há caminho de saída em caso de erro

    # Calcula tempo decorrido
    elapsed_time = time.time() - start_time
    if output_path: # Calcula tempo apenas se sucesso
        h, r = divmod(elapsed_time, 3600)
        m, s = divmod(r, 60)
        time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    else:
        time_str = "N/A" # Não mostra tempo se falhou

    return output_path, time_str, message


# =====================================
# Ponto de Entrada Principal
# =====================================
if __name__ == "__main__":
    # Cria diretório temporário se não existir (para uploads)
    # Não é ideal criar aqui, melhor usar tempfile, mas mantendo estrutura similar
    if not os.path.exists("temp_dir"):
        try:
            os.makedirs("temp_dir")
        except OSError as e:
            print(f"Aviso: Não foi possível criar temp_dir: {e}")

    # Roda a aplicação Streamlit
    main()