# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    Este é o rastreador de múltiplos alvos.

    Parâmetros
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        Uma métrica de distância para associação entre detecção e track.
    max_age : int
        Número máximo de falhas consecutivas antes de um track ser excluído.
    n_init : int
        Número de detecções consecutivas antes de um track ser confirmado. O
        estado do track é definido como `Deleted` se ocorrer uma falha nas
        primeiras `n_init` frames.

    Atributos
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A métrica de distância usada para associação entre detecção e track.
    max_age : int
        Número máximo de falhas consecutivas antes de um track ser excluído.
    n_init : int
        Número de frames em que um track permanece no estado de inicialização.
        Se houver falha nos primeiros `n_init` frames, o estado do track é definido
        como `Deleted`.
    kf : kalman_filter.KalmanFilter
        Um filtro de Kalman para rastrear trajetórias de alvos no espaço da imagem.
    tracks : List[Track]
        A lista de tracks ativos no instante atual.


    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter() # Instancia o filtro de Kalman.
        self.tracks = []   # Lista para armazenar os tracks ativos.
        self._next_id = 1  # Próximo ID a ser atribuído a um track.
 
    def predict(self):
        """Propaga as distribuições de estado dos tracks para o próximo instante.

        Esta função deve ser chamada uma vez a cada frame antes de `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Executa a atualização das medições e gerencia os tracks.

        Parâmetros
        ----------
        detections : List[deep_sort.detection.Detection]
            Uma lista de detecções no instante atual.
        """
        # Executa a correspondência em cascata.

        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Atualiza o conjunto de tracks.

        # 1. Para os tracks que foram associados com sucesso.
        for track_idx, detection_idx in matches:

            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        # 2. Para os tracks não associados, marca-os como perdidos.
        # Se um track estiver no estado Tentative, ele será deletado.
        # Se o tempo desde a última atualização exceder o limite, ele também será deletado.
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. Para as detecções não associadas, inicia novos tracks.
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Atualiza a lista de tracks, mantendo apenas os que não estão marcados como Deleted.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Atualiza a métrica de distância.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # Coleta as características dos tracks confirmados.
            if not track.is_confirmed():
                continue
            features += track.features # Adiciona as características ao conjunto.
            # Adiciona o ID do track correspondente.
            targets += [track.track_id for _ in track.features]
            track.features = []
        # Atualiza o conjunto de características na métrica de distância.
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # Calcula a matriz de custo usando a métrica de distância.
            cost_matrix = self.metric.distance(features, targets)
            # Aplica uma limitação de custo na matriz com base no filtro de Kalman.
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Divide os tracks em confirmados e não confirmados.

        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associa os tracks confirmados usando características de aparência.

        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associa os tracks não confirmados e os não associados anteriormente usando IoU.

        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # Tracks recentemente não associados.
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1] # Tracks que não foram atualizados recentemente.

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # Combina os resultados das duas correspondências.
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """Inicializa um novo track para uma detecção não associada."""

        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
