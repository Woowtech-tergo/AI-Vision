# vim: expandtab:ts=4:sw=4


class TrackState:
    """
     Tipo de enumeração para o estado de uma única trilha (track) de alvo. Tracks recém-criados são
    classificados como `Tentative` até que evidências suficientes sejam coletadas.
    Em seguida, o estado do track é alterado para `Confirmed`. Tracks que não estão mais ativos
    são classificados como `Deleted` para marcá-los para remoção do conjunto de tracks ativos.


    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Um único track de alvo com espaço de estado `(x, y, a, h)` e velocidades associadas,
    onde `(x, y)` é o centro da caixa delimitadora, `a` é a proporção largura/altura
    e `h` é a altura.

    Parâmetros
    ----------
    mean : ndarray
        Vetor médio da distribuição inicial do estado.
    covariance : ndarray
        Matriz de covariância da distribuição inicial do estado.
    track_id : int
        Um identificador exclusivo para o track.
    n_init : int
        Número de detecções consecutivas antes que o track seja confirmado. O estado
        do track é definido como `Deleted` se ocorrer uma falha dentro dos primeiros
        `n_init` frames.
    max_age : int
        Número máximo de falhas consecutivas antes que o estado do track seja definido
        como `Deleted`.
    feature : Optional[ndarray]
        Vetor de características da detecção de origem deste track. Se não for `None`,
        esta característica é adicionada ao cache `features`.

    Atributos
    ----------
    mean : ndarray
        Vetor médio da distribuição inicial do estado.
    covariance : ndarray
        Matriz de covariância da distribuição inicial do estado.
    track_id : int
        Um identificador exclusivo para o track.
    hits : int
        Número total de atualizações de medição.
    age : int
        Número total de frames desde a primeira ocorrência.
    time_since_update : int
        Número total de frames desde a última atualização de medição.
    state : TrackState
        O estado atual do track.
    features : List[ndarray]
        Cache de características. A cada atualização de medição, o vetor de características associado
        é adicionado a esta lista.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        # hits representa o número de vezes que o track foi associado.
        # Se o número de associações exceder n_init, o estado é alterado para Confirmed.
        self.hits = 1
        self.age = 1 # Redundante com time_since_update.
        # Incrementado a cada chamada de predict(); redefinido para 0 na chamada de update().
        self.time_since_update = 0

        self.state = TrackState.Tentative # Estado inicial é Tentative.
        # Cada track tem várias características; a lista é atualizada com cada nova associação.
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init 
        self._max_age = max_age

    def to_tlwh(self):
        """Obtém a posição atual no formato de caixa delimitadora `(top left x, top left y,
        width, height)`.

        Retorna
        -------
        ndarray
            A caixa delimitadora.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Obtém a posição atual no formato de caixa delimitadora `(min x, min y, max x,
        max y)`.

        Retorna
        -------
        ndarray
            A caixa delimitadora.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propaga a distribuição do estado para o instante atual usando o
        passo de predição do filtro de Kalman.

        Parâmetros
        ----------
        kf : kalman_filter.KalmanFilter
            O filtro de Kalman.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Executa o passo de atualização de medição do filtro de Kalman e atualiza o cache de características.

        Parâmetros
        ----------
        kf : kalman_filter.KalmanFilter
            O filtro de Kalman.
        detection : Detection
            A detecção associada.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        # Se o track foi associado em n_init frames consecutivos, altera para Confirmed.

        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Marca este track como perdido (sem associação no instante atual).
        """
        # Se o track estiver no estado Tentative e não for associado, muda para Deleted.
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            # Se o time_since_update exceder max_age, muda para Deleted.

            self.state = TrackState.Deleted 

    def is_tentative(self):
        """Retorna True se este track estiver no estado Tentative (não confirmado).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Retorna True se este track estiver no estado Confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Retorna True se este track estiver no estado Deleted e deve ser removido."""
        return self.state == TrackState.Deleted
