#coding:utf-8
'''
Data processing code for SNLI corpus.
==================
author: ChanLo
e-mail.com: chanlo@protonmail.ch
==================
...
'''

import json
import os
import re
import tarfile
import tempfile
import numpy as np
np.random.seed(1337)    # for reproducibility

from keras.utils import np_utils

class data_process(object):
    def __init__(self,filename):
        self.filename = filename
        self.LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        pass

    def get_data(self):
        data_list = list(self.get_sentence())
        premise = [sentence_1 for label,sentence_1,sentence_2 in data_list]
        hypothesis = [sentence_2 for label,sentence_1,sentence_2 in data_list]
        #print(max(len(x.split()) for x in premise))
        #print(max(len(x.split()) for x in hypothesis))
        label_mark = np.array([self.LABELS[label] for label,sentence_1,sentence_2 in data_list])
        label_mark = np_utils.to_categorical(label_mark,len(self.LABELS))
        pass

    def get_sentence(self):
        for i,line in enumerate(open(self.filename)):
            data = json.loads(line)
            label = data['gold_label']
            sentence_1 = ' '.join(self.extract_token_from_binary_parse(data['sentence1_binary_parse']))
            Sentence_1 = data['sentence1']
            sentence_2 = ' '.join(self.extract_token_from_binary_parse(data['sentence2_binary_parse']))
            Sentence_2 = data['sentence2']
            if label == '-':
                continue
            yield (label,sentence_1,sentence_2)
    
    def extract_token_from_binary_parse(self,parse):
        return parse.replace('(',' ').replace(')',' ').replace('-LRB-','(').replace('-RRB-',')').split()


if __name__ == '__main__':
    train_data = '../corpus/snli/snli_1.0_train.jsonl'
    dev_data = '../corpus/snli/snli_1.0_dev.jsonl'
    test_data = '../corpus/snli/snli_1.0_test.jsonl'
    data_process = data_process(test_data)
    data_process.get_data()