# app.py (Streamlit - Versão Refinada Baseada no Seu Código Funcional)

import streamlit as st
import os
import time
import cv2
import numpy as np
import tempfile
from pathlib import Path
import traceback

# --- Configuração da Página ---
st.set_page_config(
    page_title="AI Vision",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("👁️ AI Vision") # Título Principal

# --- Importações e Verificações de Módulos ---
# Garante que a aplicação não quebre totalmente se um módulo faltar
try:
    from ultralytics import YOLO
    import deep_sort.deep_sort.deep_sort as ds # Necessário se yolo_deepsort o usar internamente
    from Modelos.YOLOv8DeepSortTracking import yolo_deepsort
    from Modelos.ContadorDePessoasEmVideo.Person import Person
    MODELS_AVAILABLE = True
    print("Módulos de modelos carregados.")
except ImportError as e:
    st.error(f"Erro Crítico ao importar módulos: {e}. Verifique a estrutura de pastas e dependências.")
    print(f"Erro Crítico nos Imports: {e}")
    MODELS_AVAILABLE = False
    # Define objetos mock para evitar NameError se os imports falharem
    class MockYoloDeepSort:
        def start_processing(*args, **kwargs): raise ImportError("YOLO/DeepSort não carregado")
        def stop_processing(*args, **kwargs): pass
        def get_detectable_classes(*args, **kwargs): return ["Erro - Modelo YOLO não carregado"]
    yolo_deepsort = MockYoloDeepSort()
    class Person: pass # Mock simples

# --- Flag Global de Controle ---
# (Ainda relevante se suas funções de processamento a usarem)
should_continue = True

# --- Funções de Processamento (do seu código funcional) ---
# Adicionando Docstrings e tratamento de erro básico

def yolo_start_processing(input_data, output_dir, detect_class, model_file="yolov8n.pt"):
    """
    Invoca a função de processamento do YOLOv8 + DeepSort do módulo importado.

    Args:
        input_data (str): Caminho para o vídeo de entrada.
        output_dir (str): Diretório para salvar o vídeo de saída.
        detect_class (str): Classe a ser detectada/rastreada.
        model_file (str): Modelo YOLO a ser usado.

    Returns:
        tuple: (caminho_saida, outro_valor) ou (None, None).
    """
    global should_continue
    should_continue = True
    if not MODELS_AVAILABLE or not hasattr(yolo_deepsort, 'start_processing'):
        raise ModuleNotFoundError("Módulo YOLO/DeepSort ou função 'start_processing' não disponível.")
    print(f"Executando YOLO: input={input_data}, output_dir={output_dir}, classe={detect_class}")
    # Chama a função do seu módulo. Adapte se a assinatura for diferente.
    # Garanta que ela use a flag global 'should_continue' ou tenha sua própria parada.
    return yolo_deepsort.start_processing(input_data, output_dir, detect_class, model_file)

def yolo_stop_processing():
    """Tenta parar o processamento YOLO/DeepSort."""
    global should_continue
    should_continue = False
    print("Tentando parar YOLO/DeepSort (flag global setada).")
    if MODELS_AVAILABLE and hasattr(yolo_deepsort, 'stop_processing'):
        try:
            yolo_deepsort.stop_processing()
        except Exception as e:
            print(f"Erro ao chamar yolo_deepsort.stop_processing: {e}")

def contador_start_processing(input_data, output_dir, detect_class=None, model_file=None):
    """
    Executa a contagem de pessoas por subtração de fundo (lógica original).

    Args:
        input_data (str): Caminho do vídeo de entrada.
        output_dir (str): Diretório de saída.
        detect_class (str, optional): Ignorado.
        model_file (str, optional): Ignorado.

    Returns:
        tuple: (caminho_saida, caminho_saida) ou (None, None).
    """
    global should_continue
    should_continue = True
    if not MODELS_AVAILABLE: # Precisa da classe Person
         raise ModuleNotFoundError("Classe 'Person' não disponível.")

    print(f"Executando Contador: input={input_data}, output_dir={output_dir}")
    if not isinstance(input_data, str) or not os.path.exists(input_data):
        print(f"Erro Contador: Caminho inválido '{input_data}'")
        return None, None

    cap = None
    out = None
    output_video_path = None # Inicializa para o bloco finally

    try:
        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened(): raise ValueError(f"Não abriu vídeo: {input_data}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0: raise ValueError(f"Dimensões inválidas: {w}x{h}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Configurações do Contador ---
        frameArea = w * h; areaTH = frameArea * 0.003
        leftmostLine = int(1/6 * w); rightmostLine = int(5/6 * w)
        leftmostLimit = int(1/12 * w); rightmostLimit = int(11/12 * w)
        leftmostLineColor = (255, 0, 0); rightmostLineColor = (0, 0, 255)
        pt1 = [rightmostLine, 0]; pt2 = [rightmostLine, h]; pts_L1 = np.array([pt1, pt2], np.int32)
        pt3 = [leftmostLine, 0]; pt4 = [leftmostLine, h]; pts_L2 = np.array([pt3, pt4], np.int32)
        pt5 = [leftmostLimit, 0]; pt6 = [leftmostLimit, h]; pts_L3 = np.array([pt5, pt6], np.int32)
        pt7 = [rightmostLimit, 0]; pt8 = [rightmostLimit, h]; pts_L4 = np.array([pt7, pt8], np.int32)
        backgroundSubtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernelOp = np.ones((3,3),np.uint8); kernelCl = np.ones((9,9),np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        persons = []; max_p_age = 5; pid = 1
        leftCounter = 0; rightCounter = 0
        # --- Fim Configurações ---

        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"contador_{Path(input_data).stem}.mp4"
        output_video_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        if not out.isOpened(): raise ValueError("Não abriu VideoWriter")

        prog_text = "Processando Contador..."
        prog_bar = st.progress(0, text=prog_text)
        frame_count = 0

        while cap.isOpened() and should_continue:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # --- Lógica do Frame (igual à sua versão funcional) ---
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
                    cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                    x,y,w_box,h_box = cv2.boundingRect(cnt)
                    newPerson = True
                    if leftmostLimit < cx < rightmostLimit: # Checando limites X
                        for person in persons:
                            tolerance = w_box * 0.3
                            if abs(cx - person.getX()) <= (w_box / 2 + tolerance) and \
                               abs(cy - person.getY()) <= (h_box / 2 + tolerance):
                                newPerson = False
                                person.updateCoords(cx, cy)
                                current_pids.append(person.getId())
                                if person.getState()=='0':
                                    if person.goingLeft(rightmostLine,leftmostLine): leftCounter += 1
                                    elif person.goingRight(rightmostLine,leftmostLine): rightCounter += 1
                                break
                        if newPerson == True:
                            p = Person(pid, cx, cy, max_p_age)
                            persons.append(p)
                            current_pids.append(pid)
                            pid += 1
                    cv2.rectangle(frame,(x,y),(x+w_box,y+h_box),(0,255,0),1)
                    cv2.circle(frame,(cx,cy), 3, (0,0,255), -1)

            persons_to_remove = []
            for person in persons:
                person_x = person.getX()
                if person.getState() == '1':
                    if person.getDir() == 'right' and person_x > rightmostLimit: person.setDone()
                    elif person.getDir() == 'left' and person_x < leftmostLimit: person.setDone()
                if person.timedOut() and person.getId() not in current_pids:
                    persons_to_remove.append(person)
            for p_rem in persons_to_remove:
                try: persons.remove(p_rem)
                except ValueError: pass
                del p_rem

            leftMsg = f'Esq: {leftCounter}'; rightMsg = f'Dir: {rightCounter}'
            cv2.polylines(frame,[pts_L1],False,rightmostLineColor,thickness=1)
            cv2.polylines(frame,[pts_L2],False,leftmostLineColor,thickness=1)
            cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
            cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
            cv2.putText(frame,leftMsg ,(10,20),font,0.6,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,leftMsg ,(10,20),font,0.6,leftmostLineColor,1,cv2.LINE_AA)
            cv2.putText(frame,rightMsg,(10,45),font,0.6,(0,0,0),3,cv2.LINE_AA); cv2.putText(frame,rightMsg,(10,45),font,0.6,rightmostLineColor,1,cv2.LINE_AA)
            # --- Fim Lógica Frame ---
            out.write(frame)
            # Atualiza barra de progresso
            if total_frames > 0:
                percent_done = int(100 * frame_count / total_frames)
                prog_bar.progress(percent_done, text=f"{prog_text} {percent_done}%")

        # Fim do loop
        prog_bar.progress(100, text="Contador Concluído!")
        print(f"Contador: Processamento finalizado. Saída: {output_video_path}")
        return output_video_path, output_video_path # Retorna duas vezes

    except Exception as e:
        error_message = f"Erro durante o processamento do Contador: {e}"
        st.error(error_message)
        print(error_message)
        traceback.print_exc()
        return None, None # Indica falha
    finally:
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()
        cv2.destroyAllWindows()
        print("Recursos do OpenCV liberados (Contador).")


def contador_stop_processing():
    """Seta a flag para parar o loop do contador."""
    global should_continue
    should_continue = False
    print("Tentativa de parada do Contador (flag global setada).")

# --- Dicionário de Funções (como no seu exemplo) ---
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
        "get_detectable_classes": lambda mf: ["person"] # Contador não seleciona classe
    }
}

def get_detect_classes(model_name):
    """Busca classes detectáveis no dicionário de modelos."""
    info = model_functions.get(model_name)
    # Verifica se a função existe *antes* de chamá-la
    if info and "get_detectable_classes" in info and callable(info["get_detectable_classes"]):
        try:
            classes = info["get_detectable_classes"](info.get("model_file"))
            return classes if isinstance(classes, list) else []
        except Exception as e:
            print(f"Erro em get_detectable_classes para {model_name}: {e}")
            return []
    return [] # Retorna lista vazia se não aplicável


# =====================================
# Função Helper para Orquestrar Processamento
# =====================================
def run_model_processing(temp_input_path, model_name, selected_class):
    """
    Chama a função de processamento correta e lida com o resultado.

    Args:
        temp_input_path (str): Caminho do arquivo de vídeo temporário.
        model_name (str): Nome do modelo selecionado.
        selected_class (str): Classe selecionada.

    Returns:
        tuple: (caminho_saida, tempo_str, mensagem_status) ou (None, None, mensagem_erro).
    """
    start_time = time.time()
    output_path = None
    status_message = "Erro: Falha ao iniciar processamento."
    elapsed_time_str = "N/A"

    model_info = model_functions.get(model_name)
    if not model_info or "start_processing" not in model_info:
        status_message = f"Erro: Configuração do modelo '{model_name}' não encontrada."
        print(status_message)
        return None, elapsed_time_str, status_message

    start_func = model_info["start_processing"]
    output_dir = os.path.join(os.getcwd(), "outputs_st") # Diretório de saída para Streamlit

    try:
        # Chama a função específica (YOLO ou Contador)
        # Elas devem retornar (caminho_saida, outro_valor) ou (caminho_saida, caminho_saida)
        temp_output_path, _ = start_func(
            temp_input_path,
            output_dir,
            selected_class,
            model_info.get("model_file")
        )

        if temp_output_path and os.path.exists(temp_output_path):
            output_path = temp_output_path # Guarda o caminho válido
            status_message = "Processamento concluído com sucesso!"
            print(f"Sucesso: {status_message} Saída: {output_path}")
        else:
            status_message = "Erro: Processamento finalizado, mas não gerou arquivo de saída."
            print(f"{status_message} Caminho retornado: {temp_output_path}")
            output_path = None

    except ModuleNotFoundError as e:
        status_message = f"Erro de Importação: {e}. Verifique as dependências e caminhos."
        st.error(status_message) # Mostra erro na UI também
        print(status_message); traceback.print_exc()
        output_path = None
    except Exception as e:
        status_message = f"Erro inesperado durante processamento '{model_name}': {e}"
        st.error(status_message) # Mostra erro na UI também
        print(status_message); traceback.print_exc()
        output_path = None

    # Calcula tempo apenas se o processamento foi bem-sucedido
    if output_path:
        elapsed_time = time.time() - start_time
        h, r = divmod(elapsed_time, 3600)
        m, s = divmod(r, 60)
        elapsed_time_str = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    return output_path, elapsed_time_str, status_message


# =====================================
# Interface Principal Streamlit (main)
# =====================================
def main():
    """Monta e gerencia a interface Streamlit."""

    # --- Inicialização do Estado da Sessão ---
    if 'model_name' not in st.session_state:
        st.session_state.model_name = list(model_functions.keys())[0]
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None # Será definido após seleção do modelo
    if 'uploaded_file_bytes' not in st.session_state:
        st.session_state.uploaded_file_bytes = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'output_path' not in st.session_state:
        st.session_state.output_path = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Carregue um vídeo para começar."
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'temp_input_path' not in st.session_state: # Armazena caminho temporário durante processamento
        st.session_state.temp_input_path = None


    # --- Layout ---
    col1, col2 = st.columns(2)

    # --- Coluna 1: Inputs ---
    with col1:
        st.header("1. Configuração e Pré-visualização")

        uploaded_file = st.file_uploader(
            "Carregar Vídeo",
            type=["mp4", "avi", "mov", "mkv"],
            key="fileuploader",
            help="Selecione o vídeo a ser processado."
        )

        # Lógica para lidar com o upload e exibir visualização
        if uploaded_file is not None:
            # Atualiza o estado apenas se for um arquivo novo ou diferente
            if uploaded_file.name != st.session_state.get('uploaded_file_name'):
                st.session_state.uploaded_file_bytes = uploaded_file.read()
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.output_path = None # Limpa resultado anterior
                st.session_state.processing_time = None
                st.session_state.status_message = "Vídeo carregado. Pronto para configurar."
                print(f"Arquivo '{uploaded_file.name}' carregado.")
                # Importante: Não salva em disco aqui, apenas no estado da sessão

            st.write("Vídeo Carregado:")
            st.video(st.session_state.uploaded_file_bytes) # Exibe usando os bytes em memória
        else:
            # Limpa o estado se nenhum arquivo estiver carregado
            if st.session_state.uploaded_file_name is not None:
                 st.session_state.uploaded_file_bytes = None
                 st.session_state.uploaded_file_name = None
                 st.session_state.output_path = None
                 st.session_state.processing_time = None
                 st.session_state.status_message = "Carregue um vídeo para começar."
                 print("Upload removido ou expirado.")
            st.info("Aguardando upload do vídeo...")


        # Seleção de Modelo (lê e escreve no estado da sessão)
        st.session_state.model_name = st.selectbox(
            "Selecione o Modelo:",
            list(model_functions.keys()),
            key='model_select',
            index=list(model_functions.keys()).index(st.session_state.model_name)
        )

        # Seleção de Classe (dependente do modelo)
        available_classes = get_detect_classes(st.session_state.model_name)
        if available_classes: # Mostra apenas se houver classes
             # Garante que o valor padrão esteja na lista ou usa o primeiro item
             if st.session_state.selected_class not in available_classes:
                  st.session_state.selected_class = available_classes[0] if available_classes else None

             current_selection_index = available_classes.index(st.session_state.selected_class) if st.session_state.selected_class in available_classes else 0
             st.session_state.selected_class = st.selectbox(
                f"Selecione a Classe ({st.session_state.model_name}):",
                available_classes,
                index=current_selection_index,
                key='class_select',
                disabled=st.session_state.is_processing # Desabilita durante processamento
             )
        else:
            st.caption(f"Classe não aplicável para {st.session_state.model_name}.")
            st.session_state.selected_class = None # Limpa se não aplicável


        # Botões
        process_button_disabled = (st.session_state.uploaded_file_bytes is None or st.session_state.is_processing)
        stop_button_disabled = not st.session_state.is_processing

        process_clicked = st.button("Processar Vídeo", key="process_btn", disabled=process_button_disabled)
        stop_clicked = st.button("Parar Processamento", key="stop_btn", disabled=stop_button_disabled)

        # Lógica de Parada (acionada pelo botão)
        if stop_clicked:
            print("Botão Parar clicado.")
            model_info = model_functions.get(st.session_state.model_name)
            if model_info and "stop_processing" in model_info:
                try:
                    model_info["stop_processing"]() # Chama a função de parada
                    st.session_state.status_message = "Sinal de interrupção enviado..."
                    st.warning(st.session_state.status_message)
                    # Nota: A parada efetiva depende do loop de processamento checar a flag
                    st.session_state.is_processing = False # Assume que a parada será efetiva (pode precisar de ajuste)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao tentar parar: {e}")
                    print(f"Erro ao chamar stop_processing: {e}")
            else:
                st.error("Função de parada não encontrada para o modelo.")

    # --- Coluna 2: Outputs ---
    with col2:
        st.header("2. Resultado")

        if st.session_state.is_processing:
            st.info("Processamento em andamento...")
            # Barra de progresso será exibida aqui se a função de processamento a usar

        elif st.session_state.output_path and os.path.exists(st.session_state.output_path):
             st.success(st.session_state.status_message)
             if st.session_state.processing_time:
                 st.info(f"Tempo de processamento: {st.session_state.processing_time}")

             st.write("Vídeo Processado:")
             try:
                 # Abre o arquivo de saída para exibição e download
                 with open(st.session_state.output_path, "rb") as f_out:
                     output_video_bytes = f_out.read()

                 st.video(output_video_bytes) # Exibe o vídeo

                 # Botão de Download
                 st.download_button(
                    label="Baixar Vídeo Processado",
                    data=output_video_bytes,
                    file_name=os.path.basename(st.session_state.output_path), # Nome do arquivo
                    mime="video/mp4",
                    key='download_btn'
                 )
             except FileNotFoundError:
                  st.error(f"Erro: Arquivo de saída não encontrado: {st.session_state.output_path}")
                  print(f"Erro: FileNotFoundError ao tentar ler {st.session_state.output_path}")
                  st.session_state.output_path = None # Limpa estado inválido
             except Exception as e:
                  st.error(f"Erro ao carregar/exibir vídeo processado: {e}")
                  print(f"Erro ao carregar/exibir {st.session_state.output_path}: {e}")
                  traceback.print_exc()
                  st.session_state.output_path = None # Limpa estado inválido
        else:
            # Mensagem padrão se não estiver processando e não houver resultado
            st.info(st.session_state.status_message)


    # --- Lógica de Processamento (Disparada pelo Botão 'Processar') ---
    if process_clicked and st.session_state.uploaded_file_bytes:
        # 1. Sinalizar início e limpar estado anterior
        st.session_state.is_processing = True
        st.session_state.output_path = None
        st.session_state.processing_time = None
        st.session_state.status_message = f"Preparando para processar com {st.session_state.model_name}..."
        print(f"Processamento iniciado pelo usuário: Modelo={st.session_state.model_name}, Classe={st.session_state.selected_class}")

        # 2. Criar arquivo temporário para o vídeo carregado
        temp_dir = tempfile.mkdtemp(prefix="st_vid_") # Diretório temporário seguro
        temp_input_path = os.path.join(temp_dir, st.session_state.uploaded_file_name)
        st.session_state.temp_input_path = temp_input_path # Guarda para limpeza posterior
        try:
            with open(temp_input_path, "wb") as f:
                f.write(st.session_state.uploaded_file_bytes)
            print(f"Vídeo de entrada salvo temporariamente em: {temp_input_path}")

            # 3. Disparar o rerun para mostrar o status "Processando..."
            # A lógica de chamada da função pesada virá APÓS este rerun.
            st.rerun()

        except Exception as e:
            st.session_state.status_message = f"Erro ao salvar vídeo temporário: {e}"
            st.error(st.session_state.status_message)
            print(st.session_state.status_message); traceback.print_exc()
            st.session_state.is_processing = False # Cancela se não conseguiu salvar
            # Tenta limpar se o diretório foi criado
            if os.path.exists(temp_dir):
                try:
                    if os.path.exists(temp_input_path): os.remove(temp_input_path)
                    os.rmdir(temp_dir)
                except OSError: pass
            st.session_state.temp_input_path = None


    # 4. Executar o processamento pesado (APÓS o rerun, quando is_processing é True)
    if st.session_state.is_processing and st.session_state.temp_input_path:
        temp_input_to_process = st.session_state.temp_input_path # Pega o caminho salvo

        # Mostra spinner enquanto processa
        with st.spinner(f"Executando {st.session_state.model_name}... Isso pode levar um tempo."):
             output_path, elapsed_time_str, message = run_model_processing(
                temp_input_to_process,
                st.session_state.model_name,
                st.session_state.selected_class
             )

        # Atualiza estado com resultados
        st.session_state.output_path = output_path
        st.session_state.processing_time = elapsed_time_str
        st.session_state.status_message = message

        # Limpa arquivo/diretório temporário de input
        if os.path.exists(temp_input_to_process):
             try:
                 temp_dir_to_remove = os.path.dirname(temp_input_to_process)
                 os.remove(temp_input_to_process)
                 os.rmdir(temp_dir_to_remove)
                 print(f"Arquivo/Diretório temporário de input removido: {temp_dir_to_remove}")
             except OSError as e:
                 print(f"Aviso: Falha ao remover diretório/arquivo temporário {temp_input_to_process}: {e}")
        st.session_state.temp_input_path = None # Limpa do estado

        # Finaliza estado de processamento e dispara rerun final para mostrar resultados
        st.session_state.is_processing = False
        st.rerun()


# --- Ponto de Entrada ---
if __name__ == "__main__":
    # Garante que o diretório de saída principal existe
    main_output_dir = os.path.join(os.getcwd(), "outputs_st")
    if not os.path.exists(main_output_dir):
        os.makedirs(main_output_dir)
        print(f"Diretório de saída criado: {main_output_dir}")

    # Executa a função principal que monta a UI
    main()