import gdown
import os
from tensorflow.keras.utils import get_file


def download_oxford_dataset(path='../dataset'):
    # oxford dataset
    url = 'https://drive.google.com/uc?id=1dIR9ANjUsV9dWa0pS9J0c2KUGMfpIRG0'
    fname = 'oxford_pet.zip'
    path = os.path.join(path, fname)
    gdown.download(url, path, quiet=False)


def download_VGGNet_weights_Topless():
    ## vgg16 pretrained weights 다운로드
    return get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                           'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
                           'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
