"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import mlrose
import numpy as np


class MIMIC:
    def __init__(self, pop_size=300, keep_pct=0.2, max_attempts=45,
                 max_iters=np.inf, curve=False, random_state=None, fast_mimic=False):
        print("Initialized MIMIC Optimizer")
        self.pop_size = pop_size
        self.keep_pct = keep_pct
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.curve = curve
        self.random_state = random_state
        self.fast_mimic = fast_mimic
        self.bestState = []
        self.bestFitness = 0
        self.parameters = {'max_iters': np.arange(1, 251),
                           'pop_size': np.arange(100, 400, 100),
                           'keep_pct': [0.1, 0.2, 0.3]}
        self.bestParameters = {'max_iters': int(max(np.arange(1, 251))), 'pop_size': 300, 'keep_pct': 0.1}

    def getBestStateAndFit(self):
        return self.bestState, self.bestFitness, self.bestParameters

    def getOptimizerName(self):
        return 'MIMIC'
