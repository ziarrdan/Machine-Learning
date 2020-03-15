"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import sklearn.model_selection as ms
import numpy as np
import pandas as pd
import random


# Based on https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
def is_balanced(seq):
    n = len(seq)
    classes = [(clas, float(count)) for clas, count in Counter(seq).items()]
    k = len(classes)

    H = -sum([(count/n) * np.log((count/n)) for clas, count in classes])
    return H/np.log(k) > 0.75

# The parent dataset member variables and functions are chosen based on
# https://github.com/martzcodes/machine-learning-assignments/blob/master/supervised-learning/data/loader.py
class dataset:
    def __init__(self, path=None, verbose=None, seed=None):
        self.path = path
        self.verbose = verbose
        self.seed = seed

        self.features = None
        self.classes = None
        self.testing_x = []
        self.testing_y = []
        self.training_x = []
        self.training_y = []
        self.binary = False
        self.balanced = False
        self.data = pd.DataFrame()

    def get_features(self, force=False):
        if self.features is None or force:
            self.features = np.array(self.data.iloc[:, 0:-1])

        return self.features

    def get_classes(self, force=False):
        if self.classes is None or force:
            self.classes = np.array(self.data.iloc[:, -1])

        return self.classes

    def scale_standard(self):
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=6):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self.seed, stratify=self.classes)

    def build_train_test_splitSecond(self, test_size=0.2):
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.training_x, self.training_y, test_size=test_size, random_state=self.seed, stratify=self.training_y)

    def processDataset(self):
        self.build_train_test_split()
        self.scale_standard()

class wineDS(dataset):
    def __init__(self, path='./Datasets/winequality-red.csv', verbose=False, seed=1):
        # adapted from https://www.kaggle.com/muammerhuseyinoglu/prediction-of-wine-quality
        super().__init__(path, verbose, seed)
        random.seed(seed)
        np.random.seed(seed)
        self.name = 'Wine Quality Dataset'
        self.datasetNo = 2
        self.data = pd.read_csv(self.path, header=0)
        self.data = self.data[:1200]
        self.data = self.data.sample(frac=1)
        self.features = self.data.iloc[:, :-1].values
        self.classes = self.data.iloc[:, -1].values
        labelencoderY = LabelEncoder()
        self.classes = labelencoderY.fit_transform(self.classes)
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)
        self.processDataset()

class incomeDS(dataset):
    def __init__(self, path='./Datasets/census-reproduced.csv', verbose=False, seed=1):
        # the preprocessed dataset is downloaded from
        # https://github.com/Heronwang/GATECH-CS7641-Machine-Learning/tree/master/Assignment3
        super().__init__(path, verbose, seed)
        random.seed(seed)
        np.random.seed(seed)
        self.name = 'Income Dataset'
        self.datasetNo = 1
        self.data = pd.read_csv(self.path, header=0)
        self.data = self.data.sample(frac=1)
        self.data = self.data[:2000]
        self.features = self.get_features()
        self.classes = self.get_classes()
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)
        self.balanced = False  # Data distribution is almost 75-25%
        self.build_train_test_split()
