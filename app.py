# app.py (Streamlit - Versão Refinada com Logging Adicional)

import streamlit as st
import os
import time
import cv2
import numpy as np
import tempfile
from pathlib import Path
import traceback
import logging # Importa o módulo de logging

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Iniciando script app.py...")

# --- Configuração da Página ---
try:
    st.set_page_config(
        page_title="AI Vision",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    logging.info("Configuração da página Streamlit aplicada.")
except Exception as e:
    # Pode falhar se chamado depois de outros comandos st, mas tentamos logo
    logging.warning(f"Não foi possível configurar a página (pode já ter sido configurada): {e}")

st.title("👁️ AI Vision")
logging.info("Título da página definido.")

# --- Importações e Verificações ---
MODELS_AVAILABLE = False # Assume Falso inicialmente
try:
    logging.info("Tentando importar Ultralytics...")
    from ultralytics import YOLO
    logging.info("Importou Ultralytics.")
    logging.info("Tentando importar DeepSort...")
    import deep_sort.deep_sort.deep_sort as ds
    logging.info("Importou DeepSort.")
    logging.info("Tentando importar Modelos/YOLO...")
    from Modelos.YOLOv8DeepSortTracking import yolo_deepsort
    logging.info("Importou Modelos/YOLO.")
    logging.info("Tentando importar Modelos/Contador...")
    from Modelos.ContadorDePessoasEmVideo.Person import Person
    logging.info("Importou Modelos/Contador.")
    MODELS_AVAILABLE = True
    logging.info("Todos os módulos de modelos foram carregados com sucesso.")
except ImportError as e:
    error_msg = f"Erro Crítico ao importar módulos: {e}. Verifique estrutura/dependências."
    st.error(error_msg)
    logging.error(error_msg)
    # Define mocks para evitar NameError posterior
    class MockYoloDeepSort:
        def start_processing(*args, **kwargs): raise ImportError("YOLO/DeepSort não carregado")
        def stop_processing(*args, **kwargs): pass
        def get_detectable_classes(*args, **kwargs): return ["Erro - Modelo YOLO não carregado"]
    yolo_deepsort = MockYoloDeepSort()
    class Person: pass
except Exception as e:
    error_msg = f"Erro inesperado durante imports: {e}"
    st.error(error_msg)
    logging.error(error_msg)
    traceback.print_exc()


# --- Flag Global ---
should_continue = True
logging.info("Flag 'should_continue' inicializada.")

# --- Funções de Processamento ---
# Adicionado logging dentro das funções

def yolo_start_processing(input_data, output_dir, detect_class, model_file="yolov8n.pt"):
    global should_continue
    should_continue = True
    logging.info(f"Iniciando yolo_start_processing: input='{input_data}', output_dir='{output_dir}', classe='{detect_class}'")
    if not MODELS_AVAILABLE or not hasattr(yolo_deepsort, 'start_processing'):
        logging.error("Erro yolo_start_processing: Módulo/Função não disponível.")
        raise ModuleNotFoundError("Módulo YOLO/DeepSort ou função 'start_processing' não disponível.")

    try:
        # Garanta que a função real exista e tenha a assinatura esperada
        output_path, other_val = yolo_deepsort.start_processing(input_data, output_dir, detect_class, model_file)
        logging.info(f"yolo_deepsort.start_processing retornou: {output_path}")
        if output_path and os.path.exists(output_path):
            logging.info("Processamento YOLO concluído com sucesso.")
            return output_path, other_val
        else:
            logging.error(f"Erro yolo_start_processing: Arquivo de saída não encontrado ou inválido: {output_path}")
            st.error("Processamento YOLOv8 concluído, mas arquivo de saída não foi encontrado.")
            return None, None
    except Exception as e:
        logging.error(f"Erro EXCEÇÃO em yolo_start_processing: {e}")
        traceback.print_exc() # Loga o traceback completo no console do servidor
        st.error(f"Erro durante a execução do YOLOv8 + DeepSort: {e}")
        return None, None

def yolo_stop_processing():
    global should_continue
    should_continue = False
    logging.info("Tentando parar YOLO/DeepSort (flag global setada).")
    if MODELS_AVAILABLE and hasattr(yolo_deepsort, 'stop_processing'):
        try:
            logging.info("Chamando yolo_deepsort.stop_processing()...")
            yolo_deepsort.stop_processing()
            logging.info("yolo_deepsort.stop_processing() chamado.")
        except Exception as e:
            logging.error(f"Erro ao chamar yolo_deepsort.stop_processing: {e}")

def contador_start_processing(input_data, output_dir, detect_class=None, model_file=None):
    global should_continue
    should_continue = True
    logging.info(f"Iniciando contador_start_processing: input='{input_data}', output_dir='{output_dir}'")
    if not MODELS_AVAILABLE:
         logging.error("Erro contador_start_processing: Classe Person não disponível.")
         raise ModuleNotFoundError("Classe 'Person' não disponível.")

    if not isinstance(input_data, str) or not os.path.exists(input_data):
        logging.error(f"Erro Contador: Caminho inválido '{input_data}'")
        return None, None

    cap = None
    out = None
    output_video_path = None

    try:
        logging.info(f"Abrindo vídeo: {input_data}")
        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened(): raise ValueError(f"Não abriu vídeo: {input_data}")
        logging.info("Vídeo aberto com sucesso.")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0: raise ValueError(f"Dimensões inválidas: {w}x{h}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Propriedades do vídeo: {w}x{h} @ {fps:.2f} FPS, Total Frames: {total_frames if total_frames > 0 else 'N/A'}")

        # --- Configurações ---
        frameArea = w * h; areaTH = frameArea * 0.003
        leftmostLine = int(1/6 * w); rightmostLine = int(5/6 * w)
        leftmostLimit = int(1/12 * w); rightmostLimit = int(11/12 * w)
        leftmostLineColor = (255, 0, 0); rightmostLineColor = (0, 0, 255)
        pt1=[rightmostLine,0];pt2=[rightmostLine,h];pts_L1=np.array([pt1,pt2],np.int32)
        pt3=[leftmostLine,0];pt4=[leftmostLine,h];pts_L2=np.array([pt3,pt4],np.int32)
        pt5=[leftmostLimit,0];pt6=[leftmostLimit,h];pts_L3=np.array([pt5,pt6],np.int32)
        pt7=[rightmostLimit,0];pt8=[rightmostLimit,h];pts_L4=np.array([pt7,pt8],np.int32)
        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernelOp=np.ones((3,3),np.uint8);kernelCl=np.ones((9,9),np.uint8)
        font=cv2.FONT_HERSHEY_SIMPLEX
        persons = [];max_p_age = 5;pid = 1;leftCounter = 0;rightCounter = 0
        logging.info("Configurações do contador inicializadas.")
        # --- Fim Configurações ---

        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"contador_{Path(input_data).stem}.mp4"
        output_video_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Tente 'avc1' ou 'X264' se mp4v falhar
        logging.info(f"Preparando VideoWriter para: {output_video_path} com fourcc 'mp4v'")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not out.isOpened(): raise ValueError("Não abriu VideoWriter")
        logging.info("VideoWriter aberto com sucesso.")

        prog_text = "Processando Contador..."
        try:
            # Placeholders para barra de progresso
            prog_bar_placeholder = st.empty()
            prog_bar_placeholder.progress(0, text=prog_text)
            logging.info("Barra de progresso inicializada.")
        except Exception as e:
             logging.warning(f"Não foi possível criar barra de progresso (pode ser rerun): {e}")
             prog_bar_placeholder = None # Evita erro se não puder criar

        frame_count = 0
        logging.info("Iniciando loop de processamento de frames...")

        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret:
                logging.info(f"Fim do vídeo ou erro de leitura no frame {frame_count}.")
                break
            frame_count += 1
            if frame_count % int(fps) == 0: # Loga a cada segundo aprox.
                 logging.info(f"Processando frame {frame_count}...")

            # --- Lógica do Frame ---
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
                    M = cv2.moments(cnt);
                    if M['m00'] == 0: continue
                    cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                    x,y,w_box,h_box = cv2.boundingRect(cnt)
                    newPerson = True
                    if leftmostLimit < cx < rightmostLimit:
                        for person in persons:
                            tolerance = w_box * 0.3
                            if abs(cx - person.getX()) <= (w_box / 2 + tolerance) and \
                               abs(cy - person.getY()) <= (h_box / 2 + tolerance):
                                newPerson = False; person.updateCoords(cx, cy); current_pids.append(person.getId())
                                if person.getState()=='0':
                                    if person.goingLeft(rightmostLine,leftmostLine): leftCounter += 1
                                    elif person.goingRight(rightmostLine,leftmostLine): rightCounter += 1
                                break
                        if newPerson:
                            p = Person(pid, cx, cy, max_p_age); persons.append(p); current_pids.append(pid); pid += 1
                    cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),(0,255,0),1)
                    cv2.circle(frame,(cx,cy), 3, (0,0,255), -1)
            persons_to_remove = []
            for person in persons:
                person_x = person.getX()
                if person.getState() == '1':
                    if person.getDir() == 'right' and person_x > rightmostLimit: person.setDone()
                    elif person.getDir() == 'left' and person_x < leftmostLimit: person.setDone()
                if person.timedOut() and person.getId() not in current_pids: persons_to_remove.append(person)
            for p_rem in persons_to_remove:
                try: persons.remove(p_rem)
                except ValueError: pass
                del p_rem
            leftMsg = f'Esq: {leftCounter}'; rightMsg = f'Dir: {rightCounter}'
            cv2.polylines(frame,[pts_L1],False,rightmostLineColor,thickness=1); cv2.polylines(frame,[pts_L2],False,leftmostLineColor,thickness=1)
            cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1); cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
            cv2.putText(frame,leftMsg ,(10,20),font,0.6,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,leftMsg ,(10,20),font,0.6,leftmostLineColor,1,cv2.LINE_AA)
            cv2.putText(frame,rightMsg,(10,45),font,0.6,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,rightMsg,(10,45),font,0.6,rightmostLineColor,1,cv2.LINE_AA)
            # --- Fim Lógica Frame ---
            out.write(frame)
            # Atualiza progresso
            if total_frames > 0 and prog_bar_placeholder:
                percent_done = int(100 * frame_count / total_frames)
                prog_bar_placeholder.progress(percent_done, text=f"{prog_text} {percent_done}%")

        # Fim do loop
        logging.info("Loop de processamento de frames concluído.")
        if prog_bar_placeholder:
             prog_bar_placeholder.progress(100, text="Contador Concluído!")
        logging.info(f"Contador: Processamento finalizado. Saída: {output_video_path}")
        return output_video_path, output_video_path

    except Exception as e:
        error_message = f"Erro EXCEÇÃO durante o processamento do Contador: {e}"
        logging.error(error_message)
        traceback.print_exc() # Loga traceback completo
        st.error(error_message) # Mostra erro na UI
        return None, None
    finally:
        # Garante liberação de recursos
        if cap and cap.isOpened():
            cap.release()
            logging.info("Recurso VideoCapture (Contador) liberado.")
        if out and out.isOpened():
            out.release()
            logging.info("Recurso VideoWriter (Contador) liberado.")
        cv2.destroyAllWindows()


def contador_stop_processing():
    global should_continue
    should_continue = False
    logging.info("Tentativa de parada do Contador (flag global setada).")

# --- Dicionário de Funções ---
model_functions = {
    "YOLOv8DeepSort": {
        "start_processing": yolo_start_processing,
        "stop_processing": yolo_stop_processing,
        "model_file": "yolov8n.pt",
        "get_detectable_classes": lambda mf: yolo_deepsort.get_detectable_classes(mf) if MODELS_AVAILABLE and hasattr(yolo_deepsort, 'get_detectable_classes') else ["person", "car", "truck", "bus"]
    },
    "ContadorDePessoasEmVideo": {
        "start_processing": contador_start_processing,
        "stop_processing": contador_stop_processing,
        "model_file": None,
        "get_detectable_classes": lambda mf: ["person"]
    }
}
logging.info("Dicionário 'model_functions' definido.")

def get_detect_classes(model_name):
    logging.info(f"Buscando classes para modelo: {model_name}")
    info = model_functions.get(model_name)
    if info and "get_detectable_classes" in info and callable(info["get_detectable_classes"]):
        try:
            classes = info["get_detectable_classes"](info.get("model_file"))
            logging.info(f"Classes encontradas para {model_name}: {classes}")
            return classes if isinstance(classes, list) else []
        except Exception as e:
            logging.error(f"Erro em get_detectable_classes para {model_name}: {e}")
            return []
    logging.warning(f"Nenhuma função 'get_detectable_classes' encontrada ou aplicável para {model_name}.")
    return []


# --- Função Helper ---
def run_model_processing(temp_input_path, model_name, selected_class):
    logging.info(f"Iniciando run_model_processing: model={model_name}, class={selected_class}, input={temp_input_path}")
    start_time = time.time()
    output_path = None
    status_message = "Erro: Falha geral no processamento."
    elapsed_time_str = "N/A"

    model_info = model_functions.get(model_name)
    if not model_info or "start_processing" not in model_info:
        status_message = f"Erro: Configuração/Função 'start_processing' não encontrada para '{model_name}'."
        logging.error(status_message)
        return None, elapsed_time_str, status_message

    start_func = model_info["start_processing"]
    output_dir = os.path.join(os.getcwd(), "outputs_st")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Diretório de saída: {output_dir}")

    try:
        logging.info(f"Chamando start_func para {model_name}...")
        temp_output_path, _ = start_func(
            temp_input_path,
            output_dir,
            selected_class,
            model_info.get("model_file")
        )
        logging.info(f"start_func retornou: {temp_output_path}")

        if temp_output_path and os.path.exists(temp_output_path):
            output_path = temp_output_path
            status_message = "Processamento concluído com sucesso!"
            logging.info(f"Sucesso: {status_message} Saída: {output_path}")
        else:
            status_message = "Erro: Processamento finalizado, mas arquivo de saída inválido ou não encontrado."
            logging.error(f"{status_message} Caminho retornado: {temp_output_path}")
            output_path = None

    except ModuleNotFoundError as e:
        status_message = f"Erro Fatal de Importação: {e}. Verifique dependências."
        logging.error(status_message); traceback.print_exc()
        st.error(status_message)
        output_path = None
    except Exception as e:
        status_message = f"Erro Inesperado durante processamento '{model_name}': {e}"
        logging.error(status_message); traceback.print_exc()
        st.error(status_message)
        output_path = None

    if output_path:
        elapsed_time = time.time() - start_time
        h, r = divmod(elapsed_time, 3600); m, s = divmod(r, 60)
        elapsed_time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        logging.info(f"Tempo decorrido: {elapsed_time_str}")

    logging.info(f"run_model_processing finalizado. output_path={output_path}, time={elapsed_time_str}, message={status_message}")
    return output_path, elapsed_time_str, status_message


# --- Interface Principal Streamlit ---
def main():
    logging.info("Iniciando função main() da UI Streamlit.")

    # --- Inicialização do Estado da Sessão ---
    if 'model_name' not in st.session_state: st.session_state.model_name = list(model_functions.keys())[0]
    if 'selected_class' not in st.session_state: st.session_state.selected_class = None
    if 'uploaded_file_bytes' not in st.session_state: st.session_state.uploaded_file_bytes = None
    if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
    if 'output_path' not in st.session_state: st.session_state.output_path = None
    if 'processing_time' not in st.session_state: st.session_state.processing_time = None
    if 'status_message' not in st.session_state: st.session_state.status_message = "Carregue um vídeo para começar."
    if 'is_processing' not in st.session_state: st.session_state.is_processing = False
    if 'temp_input_path' not in st.session_state: st.session_state.temp_input_path = None
    logging.info("Estado da sessão inicializado/verificado.")

    # --- Layout ---
    col1, col2 = st.columns(2)
    logging.info("Layout de colunas criado.")

    # --- Coluna 1: Inputs ---
    with col1:
        logging.info("Renderizando Coluna 1...")
        st.header("1. Configuração e Pré-visualização")

        uploaded_file = st.file_uploader("Carregar Vídeo", type=["mp4", "avi", "mov", "mkv"], key="fileuploader")

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.get('uploaded_file_name'):
                logging.info(f"Novo arquivo detectado: {uploaded_file.name}")
                st.session_state.uploaded_file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.output_path = None; st.session_state.processing_time = None
                st.session_state.status_message = "Vídeo carregado. Pronto para configurar."
                logging.info("Estado da sessão atualizado com novo upload.")
                # Força rerun para exibir vídeo imediatamente? Pode causar loop se não gerenciado
                # st.rerun() # CUIDADO ao usar rerun aqui

            if st.session_state.uploaded_file_bytes:
                 st.write("Vídeo Carregado:")
                 st.video(st.session_state.uploaded_file_bytes)
                 logging.info("Preview do vídeo carregado exibido.")
            else:
                 # Isso não deve acontecer se uploaded_file is not None e o nome mudou
                 logging.warning("uploaded_file existe, mas bytes não estão no estado da sessão.")
                 st.info("Aguardando processamento do upload...")

        else:
            if st.session_state.uploaded_file_name is not None:
                 logging.info("Upload removido pelo usuário.")
                 st.session_state.uploaded_file_bytes = None; st.session_state.uploaded_file_name = None
                 st.session_state.output_path = None; st.session_state.processing_time = None
                 st.session_state.status_message = "Carregue um vídeo para começar."
            # Mostra apenas se não há vídeo E não está processando
            if not st.session_state.is_processing:
                 st.info("Aguardando upload do vídeo...")

        # Seleção de Modelo
        model_keys = list(model_functions.keys())
        try:
             model_idx = model_keys.index(st.session_state.model_name)
        except ValueError:
             model_idx = 0 # Volta para o primeiro se o estado for inválido
             st.session_state.model_name = model_keys[model_idx]
        st.session_state.model_name = st.selectbox("Selecione o Modelo:", model_keys, index=model_idx, key='model_select')
        logging.info(f"Modelo selecionado: {st.session_state.model_name}")

        # Seleção de Classe
        available_classes = get_detect_classes(st.session_state.model_name)
        class_selector_placeholder = st.empty() # Placeholder para o selectbox
        if available_classes:
             if st.session_state.selected_class not in available_classes:
                  st.session_state.selected_class = available_classes[0] if available_classes else None
             try:
                class_idx = available_classes.index(st.session_state.selected_class) if st.session_state.selected_class in available_classes else 0
             except ValueError:
                class_idx = 0
                st.session_state.selected_class = available_classes[class_idx] if available_classes else None

             st.session_state.selected_class = class_selector_placeholder.selectbox(
                f"Selecione a Classe ({st.session_state.model_name}):",
                available_classes, index=class_idx, key='class_select',
                disabled=st.session_state.is_processing
             )
             logging.info(f"Classe selecionada: {st.session_state.selected_class}")
        else:
            class_selector_placeholder.caption(f"Classe não aplicável para {st.session_state.model_name}.")
            st.session_state.selected_class = None
            logging.info("Dropdown de classe não aplicável/mostrado.")

        # Botões
        st.write("Ações:")
        process_button_disabled = (st.session_state.uploaded_file_bytes is None or st.session_state.is_processing)
        stop_button_disabled = not st.session_state.is_processing
        process_clicked = st.button("Processar Vídeo", key="process_btn", disabled=process_button_disabled)
        stop_clicked = st.button("Parar Processamento", key="stop_btn", disabled=stop_button_disabled)
        logging.info(f"Botões renderizados. Processar desabilitado: {process_button_disabled}, Parar desabilitado: {stop_button_disabled}")

        # Lógica de Parada
        if stop_clicked:
            logging.info("Botão Parar clicado.")
            model_info = model_functions.get(st.session_state.model_name)
            if model_info and "stop_processing" in model_info:
                try:
                    model_info["stop_processing"]()
                    st.session_state.status_message = "Sinal de interrupção enviado..."
                    st.warning(st.session_state.status_message)
                    logging.info(st.session_state.status_message)
                    st.session_state.is_processing = False # Assume parada
                    st.rerun()
                except Exception as e:
                    error_msg = f"Erro ao tentar parar: {e}"
                    st.error(error_msg); logging.error(error_msg)
            else:
                st.error("Função de parada não encontrada."); logging.error("Função stop_processing não encontrada.")
        logging.info("Fim da Coluna 1.")

    # --- Coluna 2: Outputs ---
    with col2:
        logging.info("Renderizando Coluna 2...")
        st.header("2. Resultado")

        status_placeholder = st.empty() # Placeholder para mensagens de status/tempo

        if st.session_state.is_processing:
            status_placeholder.info("Processamento em andamento...")
            logging.info("Exibindo status: Processando.")
            # Barra de progresso será atualizada pela função de processamento
        elif st.session_state.output_path and os.path.exists(st.session_state.output_path):
             status_placeholder.success(st.session_state.status_message)
             if st.session_state.processing_time:
                 st.info(f"Tempo de processamento: {st.session_state.processing_time}")

             st.write("Vídeo Processado:")
             try:
                 logging.info(f"Tentando ler e exibir vídeo de: {st.session_state.output_path}")
                 with open(st.session_state.output_path, "rb") as f_out:
                     output_video_bytes = f_out.read()
                 st.video(output_video_bytes)
                 logging.info("Vídeo processado exibido.")

                 st.download_button(
                    label="Baixar Vídeo Processado", data=output_video_bytes,
                    file_name=os.path.basename(st.session_state.output_path),
                    mime="video/mp4", key='download_btn'
                 )
                 logging.info("Botão de download renderizado.")
             except FileNotFoundError:
                  error_msg = f"Erro Crítico: Arquivo de saída não encontrado no caminho esperado: {st.session_state.output_path}"
                  st.error(error_msg); logging.error(error_msg)
                  st.session_state.output_path = None # Limpa estado
             except Exception as e:
                  error_msg = f"Erro ao carregar/exibir vídeo de saída: {e}"
                  st.error(error_msg); logging.error(error_msg); traceback.print_exc()
                  st.session_state.output_path = None # Limpa estado
        else:
            # Mensagem inicial ou após erro sem output_path
            status_placeholder.info(st.session_state.status_message)
            logging.info(f"Exibindo status inicial/padrão: {st.session_state.status_message}")
        logging.info("Fim da Coluna 2.")

    # --- Lógica de Processamento (Disparada via rerun) ---
    # Se o botão foi clicado no ciclo anterior E AINDA NÃO ESTAMOS PROCESSANDO
    if process_clicked and st.session_state.uploaded_file_bytes and not st.session_state.is_processing:
        logging.info("Iniciando sequência de processamento (após clique).")
        # 1. Marcar como processando e limpar resultados antigos
        st.session_state.is_processing = True
        st.session_state.output_path = None
        st.session_state.processing_time = None
        st.session_state.status_message = f"Preparando para processar com {st.session_state.model_name}..."
        st.session_state.temp_input_path = None # Limpa path temporário antigo

        # 2. Criar arquivo temporário
        temp_dir = None # Inicializa
        try:
            temp_dir = tempfile.mkdtemp(prefix="st_vid_")
            temp_input_path = os.path.join(temp_dir, st.session_state.uploaded_file_name)
            with open(temp_input_path, "wb") as f:
                f.write(st.session_state.uploaded_file_bytes)
            st.session_state.temp_input_path = temp_input_path # Guarda para usar no próximo rerun
            logging.info(f"Arquivo de input temporário criado: {temp_input_path}")
            # 3. Disparar rerun para iniciar o processamento real
            st.rerun()

        except Exception as e:
             st.session_state.status_message = f"Erro ao criar arquivo temporário: {e}"
             logging.error(st.session_state.status_message); traceback.print_exc()
             st.error(st.session_state.status_message)
             st.session_state.is_processing = False # Cancela
             st.session_state.temp_input_path = None
             # Tenta limpar diretório se foi criado
             if temp_dir and os.path.exists(temp_dir):
                 try: os.rmdir(temp_dir)
                 except OSError: pass


    # 4. Executar o processamento pesado (QUANDO is_processing é True no início do script)
    elif st.session_state.is_processing and st.session_state.temp_input_path:
        logging.info("Executando bloco de processamento pesado...")
        temp_input_to_process = st.session_state.temp_input_path # Pega o caminho
        temp_dir_to_remove = os.path.dirname(temp_input_to_process) # Guarda dir para limpar

        output_path, elapsed_time_str, message = None, "N/A", "Falha no processamento." # Defaults

        # Chama a função helper que executa o modelo
        # O spinner deve aparecer automaticamente aqui se a função demorar
        try:
            output_path, elapsed_time_str, message = run_model_processing(
                temp_input_to_process,
                st.session_state.model_name,
                st.session_state.selected_class
            )
        except Exception as proc_err:
             # Captura erros inesperados da própria run_model_processing
             message = f"Erro crítico na orquestração do processamento: {proc_err}"
             logging.error(message); traceback.print_exc()
             st.error(message)
             output_path = None; elapsed_time_str = "N/A"


        # Atualiza estado com resultados
        st.session_state.output_path = output_path
        st.session_state.processing_time = elapsed_time_str
        st.session_state.status_message = message

        # Limpa temporário de INPUT
        logging.info("Limpando arquivo/diretório temporário de input...")
        if os.path.exists(temp_input_to_process):
             try:
                 os.remove(temp_input_to_process)
                 os.rmdir(temp_dir_to_remove)
                 logging.info(f"Temporário de input removido: {temp_dir_to_remove}")
             except OSError as e:
                 logging.warning(f"Falha ao remover temporário de input {temp_input_to_process}: {e}")
        st.session_state.temp_input_path = None # Limpa do estado

        # Finaliza e dispara rerun para mostrar resultados
        st.session_state.is_processing = False
        logging.info("Processamento finalizado, disparando rerun para exibir resultados.")
        st.rerun()


# --- Ponto de Entrada ---
if __name__ == "__main__":
    logging.info("Script executado como principal.")
    # Garante que o diretório de saída principal existe
    main_output_dir = os.path.join(os.getcwd(), "outputs_st")
    if not os.path.exists(main_output_dir):
        try:
            os.makedirs(main_output_dir)
            logging.info(f"Diretório de saída principal criado: {main_output_dir}")
        except OSError as e:
            logging.error(f"Não foi possível criar diretório de saída {main_output_dir}: {e}")
            st.error(f"Não foi possível criar diretório de saída {main_output_dir}")

    # Executa a UI
    main()
    logging.info("Função main() concluída.")