"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.tree as tree
import numpy as np
import parentlearner
from sklearn.tree._tree import TREE_LEAF
from copy import deepcopy


# Based on https://stackoverflow.com/questions/49428469/pruning-decision-trees
def prune_index(inner_tree, index, threshold):
    pruned_clf = deepcopy(inner_tree)
    if pruned_clf.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        pruned_clf.children_left[index] = TREE_LEAF
        pruned_clf.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
    if pruned_clf.children_left[index] != TREE_LEAF:
        prune_index(pruned_clf, pruned_clf.children_left[index], threshold)
        prune_index(pruned_clf, pruned_clf.children_right[index], threshold)

    return pruned_clf


class DTLearner(parentlearner.ParentLearner):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=30,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 datasetNo=0,):
        self.learner = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            class_weight=class_weight)

        self.learnerPruned = self.learner

        if datasetNo == 1:
            parameters = {'criterion': ['gini', 'entropy'],
                          'splitter' : ['best', 'random'],
                          'min_samples_split': np.arange(2, 10, 1),
                          'max_depth': np.arange(1, 16, 1)}
            complexityParams = {'max_depth': np.arange(1, 16, 1)}
        elif datasetNo == 2:
            parameters = {'criterion': ['gini', 'entropy'],
                          'splitter': ['best', 'random'],
                          'min_samples_split': np.arange(2, 10, 1),
                          'max_depth': np.arange(1, 16, 1)}
            complexityParams = {'max_depth': np.arange(1, 16, 1)}

        super().__init__("DT", self.learner, parameters, complexityParams)

    def getLearner(self):
        return self.learner

    def getLearnerPruned(self):
        return self.learnerPruned

    def setLearner(self, learner):
        self.learner = learner

    def setLearnerPruned(self, learner):
        self.learnerPruned = learner

    def prune(self):
        self.learner.tree_ = prune_index(self.learner.tree_, 0, 5)
