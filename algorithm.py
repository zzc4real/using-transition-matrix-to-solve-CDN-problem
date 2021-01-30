"""
@Author : Zhecheng
@contact: henryzhong4@hotmail.com
@File   : algorithm.py
@Time   : 11th November, 2020
@Des    :
"""

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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = datetime.datetime.now()

def traditional_method(x_train, y_train, x_test, y_test, trans_matrix):
    """
    :param x_train:         input training data
    :param y_train:         noisy labels of x_train
    :param x_test:          test data
    :param y_test:          clear labels of test data
    :param trans_matrix
    :return: none
    """
    image_shape = x_train[0].shape
    x_train = np.ravel(x_train).reshape([-1, np.prod(image_shape)])
    x_test = np.ravel(x_test).reshape([-1, np.prod(image_shape)])
    matrix_inverse = np.linalg.inv(trans_matrix)

    # apply logistic to the training data set getting p(Yhat|X)
    logistic = LogisticRegression(random_state=True, verbose=1, max_iter=100)
    logistic.fit(x_train, y_train)
    y_pre_log = logistic.predict(x_test)
    y_pre_log_pro = logistic.predict_proba(x_test)

    # P(Y|X) = P(Yhat|X) / P(Yhat|Y)
    acc_eval(y_test, matrix_inverse, y_pre_log, y_pre_log_pro, model_name='logistic regressio')

    # apply gradient boosting to the training data set getting p(Yhat|X)
    gbdt = GradientBoostingClassifier(random_state=False, verbose=1, validation_fraction=0.2)
    gbdt.fit(x_train, y_train)
    y_pre_gbdt = gbdt.predict(x_test)
    y_pre_gbdt_pro = gbdt.predict_proba(x_test)
    acc_eval(y_test, matrix_inverse, y_pre_gbdt, y_pre_gbdt_pro, model_name='gbdt')



def acc_eval(y_test, matrix_inverse, y_pre, y_pre_pro, model_name=''):
    """
    :param y_test:          labels of test data
    :param matrix_inverse:  inverse transition matrix
    :param y_pre:           predicted labels
    :param y_pre_pro:       matching probability
    :param model_name:
    :return:
    """
    # P(Y|X) =  inverse P(Yhat|Y) * P(Yhat|X).T => matching probability of clear label
    aaa = np.dot(matrix_inverse, y_pre_pro.T)

    result_list = []
    for i in aaa.T:
        idx = int(np.argmax(i, axis=1))
        result_list.append(idx)

    acc = accuracy_score(y_test, y_pre)
    print('{0} - accuracy score implemented from noise label:'.format(model_name), acc)

    acc_trans = accuracy_score(y_test, result_list)
    print('{0} - accuracy score by transmission:'.format(model_name), acc_trans)


def build_model(num_class, image_shape, trans_matrix=None, direct='forward'):
    """
    :param num_class:   class of label value
    :param image_shape: 28,28,1
    :param trans_matrix:
    :param direct:
    :return:            CNN model
    """
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='normal',
                     input_shape=image_shape,name='conv1', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling1'))
    model.add(Dropout(0.2))

    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='normal', name='conv2', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2'))
    model.add(Dropout(0.2))

    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='normal', name='conv3', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling3'))
    model.add(Dropout(0.2))

    model.add(Flatten(name='flattened'))
    model.add(BatchNormalization(name='bn'))
    model.add(Dense(100, activation='relu', name='hidden_dense'))

    # MINIST_5 & MINIST_6 cnn dropout cases
    if (trans_matrix is not None) and (direct == ''):
        model.add(Dense(num_class, activation='softmax', name='noise_dist'))
        opt = Adam(0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model


def acc_eval_dnn(y_test, y_pre, model_name=''):
    result_list = []
    for i in y_pre:
        idx = int(np.argmax(i))
        result_list.append(idx)
    acc = accuracy_score(y_test, result_list)
    print('acc_eval_{0}:'.format(model_name), acc)


def get_layer_output(model, x, index=-3):
	"""
	get the computing result output of any layer you want, default the last layer.
    :param model: primary model
    :param x: input of primary model( x of model.predict([x])[0])
    :param index: index of target layer, i.e., layer[23]
    :return: result
    """
	layer = K.function([model.input], [model.layers[index].output])
	return layer([x])[0]


def train_eval(model, x_train, y_train, x_test, y_test, whether_matrix=None, direct='forward'):
    """
    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param whether_matrix:
    :param direct:
    :return:
    """
    if (whether_matrix is not None) and (direct == ''):
        y_train_sparse = to_categorical(np.matrix(y_train).T)
        model.fit(x_train, y_train_sparse, batch_size=128, epochs=100, shuffle=False,
                  validation_data=(x_test, to_categorical(np.matrix(y_test).T)))
        y_pre = model.predict_classes(x_test)
        aaa = accuracy_score(y_test, y_pre)
        print('non_trans_acc:', aaa)
        y_pre = get_layer_output(model, x_test, index=-2)
        acc_eval_dnn(y_test, y_pre, model_name='cnn_clean_dropout')
