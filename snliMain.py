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
import gloveUse
import tempfile
import keras
import keras.backend as K 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import merge, Dense, Input, Dropout, TimeDistributed, Lambda, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.core import *
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from attentionLSTM import AttentionLSTM

LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
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
train_data = process.get_data(train_data_path, LABELS)
dev_data = process.get_data(dev_data_path, LABELS)
test_data = process.get_data(test_data_path, LABELS)

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(train_data[0] + train_data[1])

VOCAB = len(tokenizer.word_counts) + 1

print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

to_seq = lambda x: pad_sequences(tokenizer.texts_to_sequences(x), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

train = prepare_data(train_data)
validation = prepare_data(dev_data)
test = prepare_data(test_data)

print('Build model...')
print('Vocab size =', len(tokenizer.word_counts)+1)
'''
GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
    use_glove = gloveUse.glove_use()
    embedding_matrix = use_glove.use_GloVe(GLOVE_STORE, VOCAB, EMBED_HIDDEN_SIZE, tokenizer)
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)


reverse = lambda x: K.reverse(x, 0)
reverse_output_shape = lambda input_shape: (input_shape[0], input_shape[1])

SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, 20))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

prem = SumEmbeddings(prem)
hypo = SumEmbeddings(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

revhypo = Lambda(reverse, reverse_output_shape)(hypo)
subjoint = merge([prem, hypo], mode=lambda x: x[0] - x[1],output_shape=(300,))
muljoint = merge([prem, hypo], mode='mul')
rmuljoint = merge([prem, revhypo], mode='mul')
rsubjoint = merge([prem, revhypo], mode=lambda x: x[0] - x[1],output_shape=(300,))
joint = merge([prem, subjoint, muljoint, rmuljoint, rsubjoint, hypo], mode='concat')

joint = Dropout(DP)(joint)
for i in range(3):
  joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
  joint = Dropout(DP)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(len(LABELS), activation='softmax')(joint)

model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([train[0], train[1]], train[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)
'''
model = load_model('./dnn.h5')

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))