"""
Course:         CS 7641 Assignment 4, Spring 2020
Date:           March 31st, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import hiive.mdptoolbox.example as mdpExm

class ForestMng:
    def __init__(self, states, reward_cut=4, reward_wait=2, prob_fire=0.1):
        self.states = states
        self.reward_cut = reward_cut
        self.reward_wait = reward_wait
        self.prob_fire = prob_fire
        self.P, self.R = self.init_forest()

    def init_forest(self):
        return mdpExm.forest(self.states, self.reward_wait, self.reward_cut, self.prob_fire)
