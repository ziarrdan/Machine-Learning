"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.svm as svm
import numpy as np
import parentlearner


class SVMLearner(parentlearner.ParentLearner):
    def __init__(self,
                 C=1.0,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 shrinking=True,
                 probability=False,
                 tol=1e-3,
                 cache_size=200,
                 class_weight=None,
                 verbose=False,
                 max_iter=-1,
                 decision_function_shape='ovr',
                 random_state=None,
                 datasetNo=0):
        self.learner = svm.SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)

        if datasetNo == 1:
            parameters = {'tol': np.arange(1e-8, 1e-1, 0.01),
                          'gamma': [(10**(-x)) for x in range(6, 0, -1)],
                          'C': np.arange(0.001, 20.1, 2),
                          'kernel' : ['poly', 'rbf']}
            complexityParams = {'C': np.arange(0.001, 20.1, 2),
                                'gamma': [(10**(-x)) for x in range(6, 0, -1)]}
        elif datasetNo == 2:
            parameters = {'tol': np.arange(1e-8, 1e-1, 0.01),
                          'gamma': [(10**(-x)) for x in range(4, -2, -1)],
                          'C': np.arange(0.001, 10.1, 1.0),
                          'kernel': ['poly', 'rbf']}
            complexityParams = {'C': np.arange(0.001, 10.1, 1.0),
                                'gamma': [(10**(-x)) for x in range(4, -2, -1)]}

        super().__init__("SVM", self.learner, parameters, complexityParams)

    def getLearner(self):
        return self.learner




