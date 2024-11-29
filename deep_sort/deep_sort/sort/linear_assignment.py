# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# O problema de atribuição linear também é conhecido como emparelhamento de peso mínimo em gráficos bipartidos.
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5

# min_cost_matching usa o algoritmo húngaro para resolver o problema de atribuição linear.
# Recebe como entrada o custo de distância de gate ou o custo IOU.
def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Resolve o problema de atribuição linear.

    Parâmetros
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        A métrica de distância recebe uma lista de tracks e detecções, bem como
        uma lista de N índices de tracks e M índices de detecções. A métrica deve
        retornar a matriz de custo NxM, onde o elemento (i, j) é o custo de associação
        entre o i-ésimo track nos índices de tracks fornecidos e a j-ésima detecção
        nos índices de detecções fornecidos.
    max_distance : float
        Limite de gate. Associações com custo maior que este valor são
        desconsideradas.
    tracks : List[track.Track]
        Uma lista de tracks previstos no momento atual.
    detections : List[detection.Detection]
        Uma lista de detecções no momento atual.
    track_indices : List[int]
        Lista de índices de tracks que mapeia as linhas em `cost_matrix` para os
        tracks em `tracks` (veja a descrição acima).
    detection_indices : List[int]
        Lista de índices de detecções que mapeia as colunas em `cost_matrix` para
        as detecções em `detections` (veja a descrição acima).

    Retorna
    -------
    (List[(int, int)], List[int], List[int])
        Retorna uma tupla com as seguintes três entradas:
        * Uma lista de índices de tracks e detecções correspondentes.
        * Uma lista de índices de tracks não correspondentes.
        * Uma lista de índices de detecções não correspondentes.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nada para corresponder.

    # Calcula a matriz de custo.
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # Executa o algoritmo húngaro, obtendo pares de índices atribuídos com sucesso.
    # Os índices das linhas correspondem aos tracks e os índices das colunas às detecções.
    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # Identifica as detecções não correspondentes.
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # Identifica os tracks não correspondentes.
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    # Percorre os pares de índices (track, detection) correspondentes.
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # Se o custo correspondente for maior que o limite max_distance,
        # considera-se como não correspondido.
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Executa o algoritmo de correspondência em cascata.

    Parâmetros
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        A métrica de distância recebe uma lista de tracks e detecções, bem como
        uma lista de N índices de tracks e M índices de detecções. A métrica deve
        retornar a matriz de custo NxM, onde o elemento (i, j) é o custo de associação
        entre o i-ésimo track nos índices fornecidos e a j-ésima detecção nos índices fornecidos.
    max_distance : float
        Limite de gate. Associações com custo maior que este valor são
        desconsideradas.
    cascade_depth: int
        A profundidade da cascata, deve ser definida como a idade máxima do track.
    tracks : List[track.Track]
        Uma lista de tracks previstos no momento atual.
    detections : List[detection.Detection]
        Uma lista de detecções no momento atual.
    track_indices : Optional[List[int]]
        Lista de índices de tracks que mapeia as linhas em `cost_matrix` para os
        tracks em `tracks` (veja a descrição acima). Padrão: todos os tracks.
    detection_indices : Optional[List[int]]
        Lista de índices de detecções que mapeia as colunas em `cost_matrix` para
        as detecções em `detections` (veja a descrição acima). Padrão: todas as detecções.

    Retorna
    -------
    (List[(int, int)], List[int], List[int])
        Retorna uma tupla com as seguintes três entradas:
        * Uma lista de índices de tracks e detecções correspondentes.
        * Uma lista de índices de tracks não correspondentes.
        * Uma lista de índices de detecções não correspondentes.

    """

    # Atribui os índices de tracks e detecções, se não fornecidos.
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # Inicializa as listas de correspondência e detecções não correspondentes.
    unmatched_detections = detection_indices
    matches = []
    # Itera por nível de profundidade na cascata.
    for level in range(cascade_depth):
        # Sai do loop se não houver detecções restantes.
        if len(unmatched_detections) == 0:  # No detections left
            break

        # Seleciona os índices de tracks no nível atual.
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        # Realiza a correspondência no nível atual usando min_cost_matching.
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
            
        # 步骤7：调用min_cost_matching函数进行匹配 
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l # 步骤8
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # 步骤9
    return matches, unmatched_tracks, unmatched_detections

'''
Matriz de custo com gate: restringe a matriz de custo ao calcular a distância
entre a distribuição de estado do filtro de Kalman e os valores medidos.
As distâncias na matriz de custo refletem a similaridade de aparência entre
tracks e detecções. 

Se um track tenta corresponder a duas detecções com características de aparência
muito semelhantes, erros podem ocorrer. Ao calcular a distância de Mahalanobis
entre cada detecção e o track, e aplicar um limite (`gating_threshold`), é possível
distinguir a detecção mais distante em termos de Mahalanobis, reduzindo erros
de correspondência.
'''
def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalida entradas inviáveis na matriz de custo com base nas distribuições
    de estado obtidas pelo filtro de Kalman.

    Parâmetros
    ----------
    kf : O filtro de Kalman.
    cost_matrix : ndarray
        A matriz de custo NxM, onde N é o número de índices de tracks
        e M é o número de índices de detecções, de forma que o elemento (i, j)
        é o custo de associação entre `tracks[track_indices[i]]` e
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        Uma lista de tracks previstos no momento atual.
    detections : List[detection.Detection]
        Uma lista de detecções no momento atual.
    track_indices : List[int]
        Lista de índices de tracks que mapeia as linhas em `cost_matrix` para os
        tracks em `tracks`.
    detection_indices : List[int]
        Lista de índices de detecções que mapeia as colunas em `cost_matrix` para
        as detecções em `detections`.
    gated_cost : Optional[float]
        Entradas na matriz de custo que correspondem a associações inviáveis são
        definidas como este valor. O padrão é um valor muito alto.
    only_position : Optional[bool]
        Se True, apenas as posições x e y da distribuição de estado são
        consideradas durante o gate. O padrão é False.

    Retorna
    -------
    ndarray
        Retorna a matriz de custo modificada.

    """
    # Invalida entradas inviáveis na matriz de custo com base nas distribuições
    # de estado obtidas pelo filtro de Kalman.
    gating_dim = 2 if only_position else 4 # Invalida entradas inviáveis na matriz de custo com base nas distribuições
    # de estado obtidas pelo filtro de Kalman.
    # A distância de Mahalanobis avalia quão longe a medição está da posição média
    # do track em termos de desvios padrão, considerando a incerteza na estimativa
    # do estado. O limite é calculado a partir da distribuição inversa qui-quadrado
    # para um intervalo de confiança de 95%.
    # Para um espaço de medição de 4 dimensões, o limite de Mahalanobis é 9.4877.
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        # Calcula a distância de gate entre a distribuição de estado do filtro
        # de Kalman e as medições
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
