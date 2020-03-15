"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""


class ParentLearner:
    def __init__(self, learnerType, learner, gridSearchParams={}, complexityParams={}):
        self.learnerType = learnerType
        self.learner = learner
        self.bestParams = []
        self.gridSearchParams = gridSearchParams
        self.complexityParams = complexityParams

    def getComplexityParams(self):
        return self.complexityParams

    def getGridSearchParams(self):
        return self.gridSearchParams

    def setGridSearchParams(self, key, value):
        self.gridSearchParams[key] = value

    def getLearnerType(self):
        return self.learnerType

    def setLearner(self, learner):
        self.learner = learner
