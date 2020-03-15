"""
Course:         CS 7641 Assignment 3, Spring 2020
Date:           March 6th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import dataloader
import experiments
from Clustering import getClusteringEvalPlots, getTsnePlot, calcClusterAdded
from DimReduction import calcICAPlotsAndReconsError, calcPCAPlotsAndReconsError, calcRPPlotsAndReconsError, calcFAPlotsAndReconsError


if __name__ == '__main__':
    # import datasets
    dataset = []
    dataset1 = dataloader.incomeDS()
    dataset.append(dataset1)
    dataset2 = dataloader.wineDS()
    dataset.append(dataset2)
    dataset1Test = dataloader.incomeDS()
    dataset1Test.build_train_test_splitSecond()
    dataset2Test = dataloader.wineDS()
    dataset2Test.build_train_test_splitSecond()

    # Part 1: Get clustering plots
    getClusteringEvalPlots(dataset)
    getTsnePlot(dataset)

    # Part 2: Get dimensionality reduction plots
    reducedDatasets1 = []
    reducedDatasets2 = []
    reducedDatasets1.append(dataset1Test)
    reducedDatasets2.append(dataset2Test)
    reducedDatasets = calcPCAPlotsAndReconsError(dataset)
    reducedDatasets1.append(reducedDatasets[0])
    reducedDatasets2.append(reducedDatasets[1])
    reducedDatasets = calcICAPlotsAndReconsError(dataset)
    reducedDatasets1.append(reducedDatasets[0])
    reducedDatasets2.append(reducedDatasets[1])
    reducedDatasets = calcRPPlotsAndReconsError(dataset)
    reducedDatasets1.append(reducedDatasets[0])
    reducedDatasets2.append(reducedDatasets[1])
    reducedDatasets = calcFAPlotsAndReconsError(dataset)
    reducedDatasets1.append(reducedDatasets[0])
    reducedDatasets2.append(reducedDatasets[1])
    getClusteringEvalPlots(reducedDatasets1)
    getTsnePlot(reducedDatasets1)
    getClusteringEvalPlots(reducedDatasets2)
    getTsnePlot(reducedDatasets2)

    # Part 3: Create the experiment class for NN Analysis
    '''exp = experiments.experiments()
    exp.getLearningCurveAll(reducedDatasets1)
    exp.getLearningCurveAll(reducedDatasets2)
    # get cluster-added datasets plots
    clusterAddedDatasets1 = []
    clusterAddedDatasets2 = []
    clusterAddedDatasets1.append(dataset1Test)
    clusterAddedDatasets2.append(dataset2Test)
    clusterAddedDatasets = calcClusterAdded(dataset)
    clusterAddedDatasets1.extend(clusterAddedDatasets[0])
    clusterAddedDatasets2.extend(clusterAddedDatasets[1])
    exp = experiments.experiments()
    exp.getLearningCurveAll(clusterAddedDatasets1, 1)
    exp.getLearningCurveAll(clusterAddedDatasets2, 1)'''





