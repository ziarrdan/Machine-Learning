"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.neighbors as neighbors
import numpy as np
import parentlearner


class KNNLearner(parentlearner.ParentLearner):
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 algorithm='auto',
                 leaf_size=30,
                 p=2,
                 metric='minkowski',
                 metric_params=None,
                 n_jobs=None,
                 datasetNo=0,
                 **kwargs):
        self.learner = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs)

        if datasetNo == 1:
            parameters = {'metric': ['manhattan', 'euclidean', 'chebyshev'],
                          'n_neighbors': np.arange(1, 32, 6)}
            complexityParams = {'n_neighbors': np.arange(1, 32, 6)}
        elif datasetNo == 2:
            parameters = {'metric': ['manhattan', 'euclidean', 'chebyshev'],
                          'n_neighbors': np.arange(1, 62, 12)}
            complexityParams = {'n_neighbors': np.arange(1, 62, 12)}

        super().__init__("KNN", self.learner, parameters, complexityParams)

    def getLearner(self):
        return self.learner




