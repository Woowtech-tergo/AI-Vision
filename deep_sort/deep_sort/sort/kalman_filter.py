# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Tabela para o quantil 0.95 da distribuição qui-quadrado com N graus de
liberdade (contém valores para N=1, ..., 9). Retirada da função chi2inv do MATLAB/Octave
e usada como limite para o gate de Mahalanobis.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

'''
O filtro de Kalman é dividido em duas etapas:
(1) Prever a posição do track no próximo instante,
(2) Atualizar a posição prevista com base na detecção.
'''
class KalmanFilter(object):
    """
    Um filtro de Kalman simples para rastreamento de caixas delimitadoras no espaço da imagem.

    O espaço de estado de 8 dimensões

        x, y, a, h, vx, vy, va, vh

    contém a posição do centro da caixa delimitadora (x, y), proporção a, altura h
    e suas respectivas velocidades.

    O movimento do objeto segue um modelo de velocidade constante. A localização da
    caixa delimitadora (x, y, a, h) é tomada como uma observação direta do espaço de estado
    (modelo de observação linear).

    Para cada trajetória, o KalmanFilter prevê a distribuição do estado. Cada trajetória
    registra sua própria média e variância como entrada do filtro.

    O espaço de estado de 8 dimensões [x, y, a, h, vx, vy, va, vh] contém a posição do
    centro da caixa delimitadora (x, y), proporção a, altura h e suas respectivas velocidades.
    O movimento do objeto segue um modelo de velocidade constante. A localização da
    caixa delimitadora (x, y, a, h) é tomada como uma observação direta do espaço de estado
    (modelo de observação linear).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Criar matrizes do modelo do filtro de Kalman.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # A incerteza de movimento e observação é escolhida com base na estimativa de estado
        # atual. Esses pesos controlam a quantidade de incerteza no modelo.
        # Isso é um pouco empírico.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Cria uma trajetória a partir de uma medição não associada.

        Parâmetros
        ----------
        measurement : ndarray
            Coordenadas da caixa delimitadora (x, y, a, h) com posição do centro (x, y),
            proporção a e altura h.

        Retorna
        -------
        (ndarray, ndarray)
            Retorna o vetor médio (8 dimensões) e a matriz de covariância (8x8
            dimensões) da nova trajetória. Velocidades não observadas são inicializadas
            com média 0.

        """


        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        # Traduz os objetos slice para concatenação ao longo do primeiro eixo
        mean = np.r_[mean_pos, mean_vel]

        # Inicializa o vetor médio (8 dimensões) e a matriz de covariância (8x8 dimensões)
        # com base na medição.
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Executa a etapa de previsão do filtro de Kalman.

        Parâmetros
        ----------
        mean : ndarray
            O vetor médio de 8 dimensões do estado do objeto no passo de tempo anterior.
        covariance : ndarray
            A matriz de covariância de 8x8 dimensões do estado do objeto no passo de tempo anterior.

        Retorna
        -------
        (ndarray, ndarray)
            Retorna o vetor médio e a matriz de covariância do estado previsto.
            Velocidades não observadas são inicializadas com média 0.

        """
        # O filtro de Kalman prevê o estado com base na média e covariância do passo anterior.

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # Inicializa a matriz de ruído de processo Q;
        # np.r_ concatena matrizes ao longo do primeiro eixo.

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # Atualiza o estado no tempo t: x' = Fx (1)
        mean = np.dot(self._motion_mat, mean)
        # Calcula a covariância de erro no tempo t: P' = FPF^T+Q (2)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Projeta a distribuição do estado para o espaço de medição.

        Parâmetros
        ----------
        mean : ndarray
            Vetor médio do estado (array de 8 dimensões).
        covariance : ndarray
            Matriz de covariância do estado (8x8 dimensões).

        Retorna
        -------
        (ndarray, ndarray)
            Retorna o vetor médio projetado e a matriz de covariância do estado estimado.


        """
        # Calcula a covariância no espaço de medição: HP'H^T + R.
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # R为测量过程中噪声的协方差；初始化噪声矩阵R
        innovation_cov = np.diag(np.square(std))

        # 将均值向量映射到检测空间，即 Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到检测空间，即 HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov # 公式(4)

    def update(self, mean, covariance, measurement):
        """Executa a etapa de correção do filtro de Kalman.

        Parâmetros
        ----------
        mean : ndarray
            Vetor médio previsto do estado (8 dimensões).
        covariance : ndarray
            Matriz de covariância do estado (8x8 dimensões).
        measurement : ndarray
            Vetor de medição de 4 dimensões (x, y, a, h), onde (x, y)
            é a posição do centro, a é a proporção e h é a altura da caixa delimitadora.

        Retorna
        -------
        (ndarray, ndarray)
            Retorna a distribuição do estado corrigida pela medição.

        """
        # Mapeia a média e a covariância para o espaço de medição, obtendo Hx' e S
        projected_mean, projected_cov = self.project(mean, covariance)

        # Fatoração de matriz
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # Calcula o ganho de Kalman K; equivalente a resolver a fórmula (5)
        # A fórmula (5) calcula o ganho de Kalman K, que é usado para avaliar a importância do erro de estimativa
        # A solução para o ganho do filtro de Kalman K utiliza a decomposição de Cholesky para acelerar a resolução.
        # No lado direito da fórmula (5), há o inverso de S. Se a matriz S for muito grande,
        # calcular o inverso de S consome muito tempo. Portanto, o código multiplica
        # ambos os lados da fórmula por S, transformando o problema em uma forma equivalente de AX=B.
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # y = z - Hx' (3)
        # Na fórmula (3), z é o vetor médio da detecção, que não inclui valores de mudança de velocidade,
        # ou seja, z = [cx, cy, r, h]. H é chamado de matriz de medição, que mapeia o vetor médio
        # da trajetória x' para o espaço de medição. Esta fórmula calcula o erro médio
        # entre a detecção e a trajetória.
        innovation = measurement - projected_mean

        # Atualiza o vetor médio: x = x' + Ky (6)
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # Atualiza a matriz de covariância: P = (I - KH)P' (7)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Calcula a distância de gate entre a distribuição do estado e as medições.

    Um limite de distância adequado pode ser obtido de `chi2inv95`. Se
    `only_position` for False, a distribuição qui-quadrado tem 4 graus de
    liberdade; caso contrário, 2.

    Parâmetros
    ----------
    mean : ndarray
        Vetor médio sobre a distribuição do estado (8 dimensões).
        (Média da distribuição do estado)
    covariance : ndarray
        Covariância da distribuição do estado (8x8 dimensões).
        (Covariância da distribuição do estado)
    measurements : ndarray
        Uma matriz Nx4 de N medições, cada uma no formato (x, y, a, h),
        onde (x, y) é o centro da caixa delimitadora, a é a proporção e h é a altura.
        (Matriz Nx4 com N medições no formato (x, y, a, h), onde (x, y) são as coordenadas do centro,
        a é a proporção e h é a altura.)
    only_position : Optional[bool]
        Se True, o cálculo da distância é feito apenas em relação à posição do
        centro da caixa delimitadora.

    Retorna
    -------
    ndarray
        Retorna um array de comprimento N, onde o i-ésimo elemento contém a
        distância de Mahalanobis ao quadrado entre (mean, covariance) e
        `measurements[i]`.
        (Array de comprimento N, onde o elemento i contém a distância de Mahalanobis ao quadrado
        entre (mean, covariance) e measurements[i].)

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
