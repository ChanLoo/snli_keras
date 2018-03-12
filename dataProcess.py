#coding:utf-8
'''
Data processing code for SNLI corpus.
==================
author: ChanLo
date: 2018.3.10
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

class data_process(object):
    def __init__(self,filename):
        self.filename = filename
        pass

    def get_data(self):
        
        pass

    def get_sentence(self):
        for i,line in enumerate(open(self.filename)):
            data = json.loads(line)
            label = data['gold_label']
            sentence_1 = ' '.join(self.extract_token_from_binary_parse(data['sentence1_binary_parse']))
            Sentence_1 = data['sentence1']
            sentence_2 = ' '.join(self.extract_token_from_binary_parse(data['sentence2_binary_parse']))
            Sentence_2 = data['sentence2']
            '''
            print(label)
            print(sentence_1)
            print(Sentence_1)
            print(sentence_2)
            print(Sentence_2)
            if i>=2:
                break
            '''
            yield (label,sentence_1,sentence_2)
    
    def extract_token_from_binary_parse(self,parse):
        return parse.replace('(',' ').replace(')',' ').replace('-LRB-','(').replace('-RRB-',')').split()


if __name__ == '__main__':
    train_data = '../corpus/snli/snli_1.0_train.jsonl'
    dev_data = '../corpus/snli/snli_1.0_dev.jsonl'
    test_data = '../corpus/snli/snli_1.0_test.jsonl'
    data_process = data_process(test_data)
    data_process.get_sentence()