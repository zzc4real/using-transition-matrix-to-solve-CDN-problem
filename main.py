

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import to_categorical
import keras.backend as K

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from cvxopt import matrix, solvers
import tensorflow as tf
import scipy.spatial.distance as ssd

import numpy as np
import sys
import datetime

from algorithm import traditional_method, build_model, train_eval

seed = datetime.datetime.now()

if __name__ == '__main__':
    # dataset_name is either FashionMNIST0.5, FashionMNIST0.6, CIFAR
    dataset_name = 'FashionMNIST0.5'
    # direction of CIFAR training process: forward, backward, or ''
    direction = ''
    # get the estimation of transition matrix
    whether_trans = 'mnist5'

    cifar = np.matrix([[0.6, 0.2, 0.2],
                       [0.2, 0.6, 0.2],
                       [0.2, 0.2, 0.6]], dtype=np.float32)

    # clean to noise, transition matrix
    mnist_5 = np.matrix([[0.5, 0.2, 0.3],
                         [0.3, 0.5, 0.2],
                         [0.2, 0.3, 0.5]], dtype=np.float32)

    mnist_6 = np.matrix([[0.4, 0.3, 0.3],
                         [0.3, 0.4, 0.3],
                         [0.3, 0.3, 0.4]], dtype=np.float32)

    # load data
    dataset = np.load('datasets/{0}.npz'.format(dataset_name))

    """
    Xtr: features of the training and validation data
    Ytr: noisy labels of the n instances. The shape is (n, ). {0, 1, 2}.
    Xts: features of the test data
    Yts: clean labels of the m instances.
    """
    Xtr = dataset['Xtr']  # 18000 or 15000
    Ytr = dataset['Str']  # 0, 1, 2
    Xts = dataset['Xts']  # 3000
    Yts = dataset['Yts']  # 0, 1, 2

    # logistic regression + Gradient Boosting
    if dataset_name == 'FashionMNIST0.5':
        # traditional_method(Xtr, Ytr, Xts, Yts, mnist_5)
        img_shape = (28, 28, 1)
        # build cnn training model
        dnn_eval = build_model(len(set(Ytr)), img_shape, whether_trans, direct=direction)
        # train and evaluate
        Xtr = np.expand_dims(Xtr, axis=3)
        Xts = np.expand_dims(Xts, axis=3)
        train_eval(dnn_eval, Xtr, Ytr, Xts, Yts, whether_matrix=whether_trans, direct=direction)

    elif dataset_name == 'FashionMNIST0.6':
        traditional_method(Xtr, Ytr, Xts, Yts, mnist_6)

