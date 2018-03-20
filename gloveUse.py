#coding:utf-8

import os
import numpy as np

class glove_use():
    """Use GloVe"""
    def __init__(self, GLOVE_STORE):
        self.GLOVE_STORE = GLOVE_STORE
        pass

    def use_GloVe(self):
        if not os.path.exists(self.GLOVE_STORE + '.npy')
        print('Computing GloVe')

        embedding_index = {}
        f = open('../Word2Vec/Glove/glove.840B.300d.txt')
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        f.close()
