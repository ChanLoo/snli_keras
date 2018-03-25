#coding:utf-8

import os
import numpy as np

class glove_use():
    """Use GloVe"""
    def __init__(self):
        pass

    def use_GloVe(self, GLOVE_STORE, VOCAB, EMBED_HIDDEN_SIZE, tokenizer):
        if not os.path.exists(GLOVE_STORE + '.npy'):
            print('Computing GloVe')

            embedding_index = {}
            with open('../Word2Vec/GloVe/glove.840B.300d.txt', encoding='utf8') as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_index[word] = coefs

            embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
            for word, i in tokenizer.word_index.items():
                embedding_vector = embedding_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    print('Missing from GloVe: {}'.format(word))

            np.save(GLOVE_STORE, embedding_matrix)

        print('Loading GloVe')
        embedding_matrix = np.load(GLOVE_STORE + '.npy')

        print('Total number of null word embeddings:')
        print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

        return embedding_matrix
        