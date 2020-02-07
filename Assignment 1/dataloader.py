"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.model_selection as ms
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from collections import Counter


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
    def __init__(self, path, verbose, seed):
        self.path = path
        self.verbose = verbose
        self.seed = seed

        self.features = None
        self.classes = None
        self.testing_x = None
        self.testing_y = None
        self.training_x = None
        self.training_y = None
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
        self.features = StandardScaler().fit_transform(self.features)
        if self.training_x is not None:
            self.training_x = StandardScaler().fit_transform(self.training_x)

        if self.testing_x is not None:
            self.testing_x = StandardScaler().fit_transform(self.testing_x)

    def build_train_test_split(self, test_size=0.2):
        if not self.training_x and not self.training_y and not self.testing_x and not self.testing_y:
            self.training_x, self.testing_x, self.training_y, self.testing_y = ms.train_test_split(
                self.features, self.classes, test_size=test_size, random_state=self.seed, stratify=self.classes)

    def processDataset(self):
        self.build_train_test_split()
        self.scale_standard()

class wineDS(dataset):
    def __init__(self, path='./Datasets/winequality-red.csv', verbose=False, seed=1):
        # adapted from https://www.kaggle.com/muammerhuseyinoglu/prediction-of-wine-quality
        super().__init__(path, verbose, seed)
        random.seed(seed)
        np.random.seed(seed)
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

class incomeDS(dataset):
    def __init__(self, path='./Datasets/income.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)
        random.seed(seed)
        np.random.seed(seed)
        self.datasetNo = 1
        self.data = pd.read_csv(self.path, header=0)
        self.data = self.data.sample(frac=1)
        self.data['class'] = self.data['class'].map({" >50K" : 1, " <=50K" : 0})

        to_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        label_encoder = preprocessing.LabelEncoder()
        one_hot = preprocessing.OneHotEncoder()
        df = self.data[to_encode]
        df = df.apply(label_encoder.fit_transform)
        vec_data = pd.DataFrame(one_hot.fit_transform(df[to_encode]).toarray())
        self.data = self.data.drop(to_encode, axis=1)
        df_class = self.data['class']
        self.data = self.data.drop(['class'], axis=1)
        self.data = pd.concat([self.data, vec_data], axis=1)
        self.data = pd.concat([self.data, df_class], axis=1)
        self.data = self.data.drop(0, axis=0)
        self.data = self.data.dropna(axis=0)
        self.features = self.get_features()
        self.classes = self.get_classes()
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)
        self.balanced = False  # Data distribution is almost 75-25%

class waveformDS(dataset):
    def __init__(self, path='./Datasets/waveform.csv', verbose=False, seed=1):
        super().__init__(path, verbose, seed)
        super().__init__(path, verbose, seed)
        random.seed(seed)
        np.random.seed(seed)
        self.datasetNo = 2
        self.data = pd.read_csv(self.path, header=0)

        self.features = self.get_features()
        self.classes = self.get_classes()
        class_dist = np.histogram(self.classes)[0]
        class_dist = class_dist[np.nonzero(class_dist)]

        if len(class_dist) == 2:
            self.binary = True
        self.balanced = is_balanced(self.classes)

