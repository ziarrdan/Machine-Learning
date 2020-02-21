"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import mlrose
import numpy as np


class SA:
    def __init__(self, schedule=mlrose.GeomDecay(), max_attempts=250,
                 max_iters=np.inf, init_state=None, curve=False, random_state=None):
        print("Initialized Simulated Annealing Optimizer")
        self.schedule = schedule
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.init_state = init_state
        self.curve = curve
        self.random_state = random_state
        self.bestState = []
        self.bestFitness = 0
        self.parameters = {'max_iters': np.arange(1, 251),
                           'schedule': [mlrose.GeomDecay(), mlrose.ArithDecay(), mlrose.ExpDecay()]}
        self.bestParameters = {'max_iters': int(max(np.arange(1, 251))), 'schedule': mlrose.GeomDecay()}

    def getBestStateAndFit(self):
        return self.bestState, self.bestFitness, self.bestParameters

    def getOptimizerName(self):
        return 'Simulated Annealing'
