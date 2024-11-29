# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Calcula a distância quadrada entre pares de pontos em `a` e `b`.

    Parâmetros
    ----------
    a : array_like
        Uma matriz NxM de N amostras com dimensionalidade M.
    b : array_like
        Uma matriz LxM de L amostras com dimensionalidade M.

    Retorna
    -------
    ndarray
        Retorna uma matriz de tamanho len(a), len(b), onde o elemento (i, j)
        contém a distância quadrada entre `a[i]` e `b[j]`.

    Usado para calcular a distância quadrada entre pares de pontos.
    `a` é uma matriz NxM, representando N amostras, cada uma com M valores.
    `b` é uma matriz LxM, representando L amostras, cada uma com M valores.
    Retorna uma matriz NxL, onde dist[i][j] representa a distância quadrada entre `a[i]` e `b[j]`.
    Referência: https://blog.csdn.net/frankzd/article/details/80251042

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Calcula a distância de cosseno entre pares de pontos em `a` e `b`.

    Parâmetros
    ----------
    a : array_like
        Uma matriz NxM de N amostras com dimensionalidade M.
    b : array_like
        Uma matriz LxM de L amostras com dimensionalidade M.
    data_is_normalized : Optional[bool]
        Se True, assume que as linhas em `a` e `b` são vetores normalizados.
        Caso contrário, `a` e `b` são explicitamente normalizados para comprimento 1.

    Retorna
    -------
    ndarray
        Retorna uma matriz de tamanho len(a), len(b), onde o elemento (i, j)
        contém a distância de cosseno entre `a[i]` e `b[j]`.

    Usado para calcular a distância de cosseno entre pares de pontos.
    Referência: https://blog.csdn.net/u013749540/article/details/51813922
    

    """
    if not data_is_normalized:
        # np.linalg.norm calcula a norma de um vetor, por padrão a norma L2
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T) # Distância de cosseno = 1 - similaridade de cosseno


def _nn_euclidean_distance(x, y):
    """ Função auxiliar para a métrica de distância do vizinho mais próximo (Euclidiana).

    Parâmetros
    ----------
    x : ndarray
        Uma matriz de N vetores linha (pontos de amostra).
    y : ndarray
        Uma matriz de M vetores linha (pontos de consulta).

    Retorna
    -------
    ndarray
        Um vetor de comprimento M que contém, para cada entrada em `y`,
        a menor distância Euclidiana para uma amostra em `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Função auxiliar para a métrica de distância do vizinho mais próximo (Cosseno).

    Parâmetros
    ----------
    x : ndarray
        Uma matriz de N vetores linha (pontos de amostra).
    y : ndarray
        Uma matriz de M vetores linha (pontos de consulta).

    Retorna
    -------
    ndarray
        Um vetor de comprimento M que contém, para cada entrada em `y`,
        a menor distância de cosseno para uma amostra em `x`.
    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    Uma métrica de distância do vizinho mais próximo que, para cada alvo, retorna
    a menor distância para qualquer amostra observada até o momento.

    Parâmetros
    ----------
    metric : str
        Pode ser "euclidean" ou "cosine".
    matching_threshold: float
        O limite de correspondência. Amostras com distância maior são consideradas
        uma correspondência inválida.
    budget : Optional[int]
        Se não for None, fixa o número máximo de amostras por classe.
        Remove as amostras mais antigas quando o limite (`budget`) é atingido.

    Atributos
    ----------
    samples : Dict[int -> List[ndarray]]
        Um dicionário que mapeia identidades de alvos para a lista de amostras
        observadas até o momento.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance  # Distância Euclidiana
        elif metric == "cosine":
            self._metric = _nn_cosine_distance # Distância de Cosseno
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget # Limita o número de características armazenadas
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Atualiza a métrica de distância com novos dados.

        Parâmetros
        ----------
        features : ndarray
            Uma matriz NxM de N características com dimensionalidade M.
        targets : ndarray
            Um array inteiro com as identidades dos alvos associados.
        active_targets : List[int]
            Uma lista de alvos atualmente presentes na cena.

        Adiciona novos recursos ao dicionário `samples` e remove os mais antigos
        se o `budget` for atingido.

        """
        for feature, target in zip(features, targets):


            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                # Limita o número de características por alvo ao `budget`
                self.samples[target] = self.samples[target][-self.budget:]
        
        # Filtra os alvos ativos no dicionário de amostras
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Calcula a distância entre características e alvos.

        Parâmetros
        ----------
        features : ndarray
            Uma matriz NxM de N características com dimensionalidade M.
        targets : List[int]
            Uma lista de alvos para comparar com as características fornecidas.

        Retorna
        -------
        ndarray
            Retorna uma matriz de custo de tamanho len(targets), len(features),
            onde o elemento (i, j) contém a menor distância quadrada entre
            `targets[i]` e `features[j]`.

        Calcula a distância entre `features` e `targets`, retornando uma matriz de custos.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
