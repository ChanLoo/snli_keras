#coding:utf-8
'''
Main code for SNLI corpus.
==================
author: ChanLo
e-mail.com: chanlo@protonmail.ch
==================
...
'''

import dataProcess
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

RNN = None
LAYERS = 1
USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'

train_data_path = '../corpus/snli/snli_1.0_train.jsonl'
dev_data_path = '../corpus/snli/snli_1.0_dev.jsonl'
test_data_path = '../corpus/snli/snli_1.0_test.jsonl'
process = dataProcess.data_process()
train_data = process.get_data(train_data_path)
dev_data = process.get_data(dev_data_path)
test_data = process.get_data(test_data_path)
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

VOCAB = len(tokenizer.word_counts) + 1

print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

to_seq = lambda x: pad_sequences(tokenizer.text_to_sequences(x), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

