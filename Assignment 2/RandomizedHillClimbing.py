"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import mlrose
import numpy as np


class RHC:
    def __init__(self, max_attempts=250, max_iters=np.inf,
                 restarts=0, init_state=None, curve=False, random_state=None):
        print("Initialized Randomized Hill Climbing Optimizer")
        self.max_iters = max_iters
        self.max_attempts = max_attempts
        self.restarts = restarts
        self.init_state = init_state
        self.curve = curve
        self.random_state = random_state
        self.bestState = []
        self.bestFitness = 0
        self.parameters = {'max_iters': np.arange(1, 251)}
        self.bestParameters = {'max_attempts': int(max(np.arange(1, 251)))}

    def getBestStateAndFit(self):
        return self.bestState, self.bestFitness, self.bestParameters

    def getOptimizerName(self):
        return 'Randomized Hill Climbing'
