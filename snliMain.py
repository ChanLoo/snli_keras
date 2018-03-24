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
from keras.layers import merge, Dense, Input, Dropout, TimeDistributed, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

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

GLOVE_STORE = 'precomputed_glove.weights'
if USE_GLOVE:
    use_glove = gloveUse.glove_use()
    embedding_matrix = use_glove.use_GloVe(GLOVE_STORE, VOCAB, EMBED_HIDDEN_SIZE, tokenizer)
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
    embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

reverse = lambda x: K.reverse(x, 0)
reverse_output_shape = lambda input_shape: (input_shape[0], input_shape[1])

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)

if RNN and LAYERS > 1:
  for l in range(LAYERS - 1):
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = BatchNormalization()(rnn(prem))
    hypo = BatchNormalization()(rnn(hypo))
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
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
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))