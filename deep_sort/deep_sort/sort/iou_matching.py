# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment

# Calcula o IOU (Intersection Over Union) entre dois boxes
def iou(bbox, candidates):
    """Calcula a interseção sobre a união.

    Parâmetros
    ----------
    bbox : ndarray
        Uma bounding box no formato `(top left x, top left y, width, height)`.
    candidates : ndarray
        Uma matriz de bounding boxes candidatas (uma por linha) no mesmo formato
        que `bbox`.

    Retorna
    -------
    ndarray
        A interseção sobre a união em [0, 1] entre o `bbox` e cada
        candidato. Um valor mais alto significa que uma fração maior do `bbox`
        é coberta pelo candidato.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # np.c_ traduz objetos slice para concatenação ao longo do segundo eixo.
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

# Calcula a matriz de custo de distância IOU entre tracks e detections
def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """Uma métrica de distância baseada em interseção sobre união.

    Utilizada para calcular a matriz de distância IOU entre tracks e detections

    Parâmetros
    ----------
    tracks : List[deep_sort.track.Track]
        Uma lista de tracks.
    detections : List[deep_sort.detection.Detection]
        Uma lista de detections.
    track_indices : Optional[List[int]]
        Uma lista de índices das tracks que devem ser associadas. Padrão é
        todas as `tracks`.
    detection_indices : Optional[List[int]]
        Uma lista de índices das detections que devem ser associadas. Padrão
        é todas as `detections`.

    Retorna
    -------
    ndarray
        Retorna uma matriz de custo com forma
        len(track_indices), len(detection_indices) onde a entrada (i, j) é
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
