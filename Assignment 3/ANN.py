"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.neural_network as neurons
import numpy as np
import parentlearner


class ANNLearner(parentlearner.ParentLearner):
    def __init__(self,
                 hidden_layer_sizes=(25,),
                 activation="relu",
                 solver='adam',
                 alpha=0.0001,
                 batch_size='auto',
                 learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 datasetNo=0,
                 ):
        self.learner = neurons.MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)

        if datasetNo == 1:
            parameters = {'alpha': [10**-x for x in np.arange(1,5.01,2)],
                          'momentum': sorted([x / 5 for x in range(5)] + [0.5]),
                          'learning_rate_init': sorted([(x+1)/1000 for x in range(2)]),
                          'hidden_layer_sizes': [(x,) for x in range(1, 42, 7)]}
            complexityParams = {'hidden_layer_sizes': [(x,) for x in range(1, 42, 7)]}
            self.trainingTimeParam = {'max_iter': np.arange(1, 300, 30)}
        elif datasetNo == 2:
            parameters = {'alpha': [10 ** -x for x in np.arange(1, 5.01, 2)],
                          'momentum': sorted([x / 5 for x in range(5)] + [0.5]),
                          'learning_rate_init': sorted([(x + 1) / 1000 for x in range(2)]),
                          'hidden_layer_sizes': [(x,) for x in range(1, 72, 14)]}
            complexityParams = {'hidden_layer_sizes': [(x,) for x in range(1, 72, 14)]}
            self.trainingTimeParam = {'max_iter': np.arange(1, 300, 30)}

        super().__init__("ANN", self.learner, parameters, complexityParams)

    def getLearner(self):
        return self.learner

    def getTrainingTimeParam(self):
        return self.trainingTimeParam




