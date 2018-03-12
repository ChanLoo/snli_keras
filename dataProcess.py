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

class dataProcess(object):
    def __init__(self,filename):
        self.filename = filename
        pass

    def getData(self):
        pass

    def getSentence(self):
        for i,line in enumerate(open(self.filename)):
            data = json.loads(line)
            print(i)
            print(line)
            print(data)
            if i >= 2:
                break
            

    def extractTokenFromBinaryParse(self):
        return self.parse.replace('(',' ').replace(')',' ').replace('-LRB-','(').replace('-RRB-',')').split()
    

if __name__ == '__main__':
    trainData = '../corpus/snli/snli_1.0_train.jsonl'
    devData = '../corpus/snli/snli_1.0_dev.jsonl'
    testData = '../corpus/snli/snli_1.0_test.jsonl'
    dataProcess = dataProcess(testData)
    dataProcess.getSentence()