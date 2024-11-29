# -*- coding:utf8 -*-

import os
from PIL import Image
from shutil import copyfile, copytree, rmtree, move

PATH_DATASET = './car-dataset'  # Pasta que precisa ser processada
PATH_NEW_DATASET = './car-reid-dataset'  # Pasta após o processamento
PATH_ALL_IMAGES = PATH_NEW_DATASET + '/all_images'
PATH_TRAIN = PATH_NEW_DATASET + '/train'
PATH_TEST = PATH_NEW_DATASET + '/test'

# Define a função para criar diretórios
def mymkdir(path):
    path = path.strip()  # Remove espaços em branco no início e no fim
    path = path.rstrip("\\")  # Remove o símbolo '\' no final
    isExists = os.path.exists(path)  # Verifica se o caminho existe
    if not isExists:
        os.makedirs(path)  # Se não existir, cria o diretório
        print(path + ' criado com sucesso')
        return True
    else:
        # Se o diretório já existir, não cria e informa que já existe
        print(path + ' diretório já existe')
        return False

class BatchRename():
    '''
    Renomeia em lote os arquivos de imagem em uma pasta
    '''

    def __init__(self):
        self.path = PATH_DATASET  # Indica a pasta que precisa ser processada

    # Modifica o tamanho das imagens
    def resize(self):
        for aroot, dirs, files in os.walk(self.path):
            # aroot é todos os subdiretórios (incluindo self.path) sob self.path, dirs é a lista de todas as pastas em self.path
            filelist = files  # Note que isso é apenas uma lista em um dos caminhos
            # print('list', list)

            # filelist = os.listdir(self.path)  # Obtém o caminho dos arquivos
            total_num = len(filelist)  # Obtém o número total de arquivos

            for item in filelist:
                if item.endswith('.jpg'):  # As imagens iniciais estão no formato jpg (ou se os arquivos originais estão em png ou outro formato, ajuste o formato conforme necessário)
                    src = os.path.join(os.path.abspath(aroot), item)

                    # Modifica o tamanho da imagem para largura 128 * altura 256
                    im = Image.open(src)
                    out = im.resize((128, 256), Image.ANTIALIAS)  # Redimensiona a imagem com alta qualidade
                    out.save(src)  # Salva no caminho original

    def rename(self):

        for aroot, dirs, files in os.walk(self.path):
            # aroot é todos os subdiretórios (incluindo self.path) sob self.path, dirs é a lista de todas as pastas em self.path
            filelist = files  # Note que isso é apenas uma lista em um dos caminhos
            # print('list', list)

            # filelist = os.listdir(self.path)  # Obtém o caminho dos arquivos
            total_num = len(filelist)  # Obtém o número total de arquivos

            i = 1  # Indica que a nomeação dos arquivos começa a partir de 1
            for item in filelist:
                if item.endswith('.jpg'):  # As imagens iniciais estão no formato jpg (ou se os arquivos originais estão em png ou outro formato, ajuste o formato conforme necessário)
                    src = os.path.join(os.path.abspath(aroot), item)

                    # Cria diretório de imagem com base no nome da imagem
                    dirname = str(item.split('_')[0])
                    # Cria diretório para o mesmo veículo
                    # new_dir = os.path.join(self.path, '..', 'bbox_all', dirname)
                    new_dir = os.path.join(PATH_ALL_IMAGES, dirname)
                    if not os.path.isdir(new_dir):
                        mymkdir(new_dir)

                    # Obtém o número de imagens em new_dir
                    num_pic = len(os.listdir(new_dir))

                    dst = os.path.join(os.path.abspath(new_dir),
                                       dirname + 'C1T0001F' + str(num_pic + 1) + '.jpg')
                    # O formato após o processamento também é jpg; você pode alterar para png se desejar
                    # 'C1T0001F' refere-se ao ID da câmera e índice de rastreamento, veja mars.py filenames
                    # dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')  # Neste caso, o formato de nomeação é 0000000.jpg, você pode definir o formato que desejar
                    try:
                        copyfile(src, dst)  # os.rename(src, dst)
                        print('convertendo %s para %s ...' % (src, dst))
                        i = i + 1
                    except:
                        continue
            print('total de %d para renomear & convertido %d jpgs' % (total_num, i))

    def split(self):
        # ---------------------------------------
        # Divisão em treino e teste
        images_path = PATH_ALL_IMAGES
        train_save_path = PATH_TRAIN
        test_save_path = PATH_TEST
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)
            os.mkdir(test_save_path)

        for _, dirs, _ in os.walk(images_path, topdown=True):
            for i, dir in enumerate(dirs):
                for root, _, files in os.walk(images_path + '/' + dir, topdown=True):
                    for j, file in enumerate(files):
                        if(j == 0):  # Conjunto de teste; primeira imagem de cada veículo
                            print("Número: %s  Pasta: %s  Imagem: %s classificada como conjunto de teste" % (i + 1, root, file))
                            src_path = root + '/' + file
                            dst_dir = test_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
                        else:
                            src_path = root + '/' + file
                            dst_dir = train_save_path + '/' + dir
                            if not os.path.isdir(dst_dir):
                                os.mkdir(dst_dir)
                            dst_path = dst_dir + '/' + file
                            move(src_path, dst_path)
        rmtree(PATH_ALL_IMAGES)

if __name__ == '__main__':
    demo = BatchRename()
    demo.resize()
    demo.rename()
    demo.split()
