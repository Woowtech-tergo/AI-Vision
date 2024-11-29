# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    Esta classe representa uma detecção de caixa delimitadora (bounding box) em uma única imagem.

    Parâmetros
    ----------
    tlwh : array_like
        Caixa delimitadora no formato `(x superior esquerdo, y superior esquerdo, largura, altura)`.
    confidence : float
        Pontuação de confiança do detector.
    feature : array_like
        Um vetor de características que descreve o objeto contido nesta imagem.

    Atributos
    ----------
    tlwh : ndarray
        Caixa delimitadora no formato `(x superior esquerdo, y superior esquerdo, largura, altura)`.
    confidence : ndarray
        Pontuação de confiança do detector.
    feature : ndarray | NoneType
        Um vetor de características que descreve o objeto contido nesta imagem.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Converte a caixa delimitadora para o formato `(x mínimo, y mínimo, x máximo, y máximo)`, ou seja,
        `(superior esquerdo, inferior direito)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Converte a caixa delimitadora para o formato `(x central, y central, proporção de aspecto,
        altura)`, onde a proporção de aspecto é `largura / altura`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
