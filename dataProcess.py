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
import temfile
import numpy as np
np.random.seed(1337)    # for reproducibility

class dataProcess(object):
    def __init__(self):
        pass

    def extractTokenFromBinaryParse(self):
