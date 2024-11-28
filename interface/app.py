import gradio as gr
import cv2
import numpy as np
import torch

# Importações necessárias para o modelo de contagem de pessoas
from Modelos.ContadorDePessoasEmVideo import Person
import time
from random import randint

def counting_people(video_input):
    # Verifica se o vídeo foi enviado através da interface ou se é um fluxo da webcam
    if video_input is None:
        cap = cv2.VideoCapture(0)  # Usa a webcam
    else:
        cap = cv2.VideoCapture(video_input)

    # Iniciais
    leftCounter = 0
    rightCounter = 0

    # Configurações de vídeo
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameArea = h * w
    areaTH = frameArea * 0.003

    # Linhas de detecção
    leftmostLine = int(1.0 / 6 * w)
    rightmostLine = int(5.0 / 6 * w)
    leftmostLimit = int(1.0 / 12 * w)
    rightmostLimit = int(11.0 / 12 * w)

    # Configurações de cores
    leftmostLineColor = (255, 0, 0)
    rightmostLineColor = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Subtrator de fundo
    backgroundSubtractor = cv2.createBackgroundSubtractorMOG2()

    # Kernels para morfologia
    kernelOp = np.ones((3, 3), np.uint8)
    kernelCl = np.ones((9, 9), np.uint8)

    # Variáveis
    persons = []
    max_p_age = 5
    pid = 1

    # Criação do objeto para salvar o vídeo processado
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        for per in persons:
            per.age_one()

        # Subtração de fundo
        fgmask = backgroundSubtractor.apply(frame)

        # Aplicação de threshold para remover sombras
        ret2, imBin = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        # Operações morfológicas
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

        # Encontra contornos
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > areaTH:
                # Centro de massa
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                x, y, w, h = cv2.boundingRect(cnt)

                newPerson = True
                if leftmostLimit <= cx <= rightmostLimit:
                    for person in persons:
                        if abs(x - person.getX()) <= w and abs(y - person.getY()) <= h:
                            newPerson = False
                            person.updateCoords(cx, cy)

                            if person.goingLeft(rightmostLine, leftmostLine):
                                leftCounter += 1
                            elif person.goingRight(rightmostLine, leftmostLine):
                                rightCounter += 1
                            break

                        if person.getState() == '1':
                            if person.getDir() == 'right' and person.getX() > rightmostLimit:
                                person.setDone()
                            elif person.getDir() == 'left' and person.getX() < leftmostLimit:
                                person.setDone()

                        if person.timedOut():
                            index = persons.index(person)
                            persons.pop(index)
                            del person

                    if newPerson:
                        person = Person(pid, cx, cy, max_p_age)
                        persons.append(person)
                        pid += 1

                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Informações no frame
        leftMsg = 'Left: ' + str(leftCounter)
        rightMsg = 'Right: ' + str(rightCounter)
        cv2.polylines(frame, [np.array([[rightmostLine, 0], [rightmostLine, h]])], False, rightmostLineColor, thickness=2)
        cv2.polylines(frame, [np.array([[leftmostLine, 0], [leftmostLine, h]])], False, leftmostLineColor, thickness=2)
        cv2.putText(frame, leftMsg, (10, 40), font, 0.5, leftmostLineColor, 1, cv2.LINE_AA)
        cv2.putText(frame, rightMsg, (10, 90), font, 0.5, rightmostLineColor, 1, cv2.LINE_AA)

        # Armazena o frame processado
        output_frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    # Salva o vídeo processado em um arquivo temporário
    output_path = 'output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20  # Pode ajustar conforme necessário
    height, width, layers = output_frames[0].shape
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in output_frames:
        out.write(frame)

    out.release()

    return output_path

# Configuração da interface Gradio
with gr.Blocks() as interface:
    gr.Markdown("# Contador de Pessoas em Vídeo")
    gr.Markdown("### Carregue um vídeo ou utilize a webcam para iniciar a contagem de pessoas.")

    video_input = gr.Video(label="Selecione um Vídeo ou use a Webcam", source=["upload", "webcam"])
    output_video = gr.Video(label="Vídeo Processado")

    start_button = gr.Button("Iniciar Contagem de Pessoas")

    start_button.click(
        fn=counting_people,
        inputs=video_input,
        outputs=output_video
    )

if __name__ == "__main__":
    interface.launch()
