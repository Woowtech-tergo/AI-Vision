import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

'''
Extrator de Características:
Extrai as características da bounding box correspondente, obtendo um embedding de dimensão fixa que representa essa bounding box, para ser usado no cálculo de similaridade.

O treinamento do modelo é feito de acordo com o método tradicional de ReID (Re-Identification). Ao usar a classe Extractor, a entrada é uma lista de imagens, e obtém-se as características correspondentes das imagens.
'''


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Carregando pesos de {}... Feito!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            # O intervalo de dados da imagem RGB é [0-255], precisamos primeiro usar ToTensor para dividir por 255 e normalizar para [0,1],
            # Em seguida, usar Normalize para calcular (x - mean)/std, normalizando os dados para [-1,1].
            transforms.ToTensor(),
            # mean=[0.485, 0.456, 0.406] e std=[0.229, 0.224, 0.225] são calculados a partir do conjunto de treinamento ImageNet
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. converter para float com escala de 0 a 1
            2. redimensionar para (64, 128) como no conjunto de dados Market1501
            3. concatenar em um array numpy
            3. converter para Tensor do PyTorch
            4. normalizar
        """

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    # __call__() é um método de instância muito especial. Sua função é semelhante a sobrecarregar o operador (),
    # permitindo que a instância da classe seja usada como uma função comum, usando a forma 'nome_do_objeto()'.
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
