"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.ensemble as ensemble
import sklearn.tree as tree
import numpy as np
import parentlearner


class BoostingLearner(parentlearner.ParentLearner):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None,
                 datasetNo=0,):
        self.baseLearner = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0,
            min_impurity_split=None,
            class_weight=None)
        self.learner = ensemble.AdaBoostClassifier(
            base_estimator=self.baseLearner,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state)

        if datasetNo == 1:
            parameters = {'learning_rate': [(2 ** x) / 100 for x in range(7)] + [1],
                          'algorithm' : ['SAMME', 'SAMME.R'],
                          'n_estimators': np.arange(1, 102, 10)}
            complexityParams = {'n_estimators': np.arange(1, 102, 10)}
        elif datasetNo == 2:
            parameters = {'learning_rate': [(2 ** x) / 100 for x in range(7)] + [1],
                          'algorithm': ['SAMME', 'SAMME.R'],
                          'n_estimators': np.arange(1, 102, 10)}
            complexityParams = {'n_estimators': np.arange(1, 102, 10)}

        super().__init__("Boosting", self.learner, parameters, complexityParams)
        self.best_params = None

    def getLearner(self):
        return self.learner




