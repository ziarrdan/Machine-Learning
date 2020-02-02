"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import dataloader
import KNN
import ANN
import DT
import SVM
import Boosting
import experiments


if __name__ == '__main__':
    # import datasets
    dataset = []
    dataset1 = dataloader.incomeDS()
    dataset1.processDataset()
    dataset.append(dataset1)
    dataset2 = dataloader.wineDS()
    dataset2.processDataset()
    dataset.append(dataset2)

    steps = 2
    file = open('Best Hyperparameters at Step '+str(1)+'.txt', 'w')
    file.write("Best Hyperparameters at Step " + str(1) + '\n')
    file.write("Figures are from step "+str(steps)+'\n')

    # create the experiment class
    exp = experiments.experiments()

    for set in dataset:
        file.write("Dataset "+str(set.datasetNo)+" Information:\n")
        file.flush()

        # start the first learner KNN
        knn = KNN.KNNLearner(datasetNo=set.datasetNo)
        _, bestParams = exp.getBestGridSearchedModel(knn, set)
        print("Best Grid-Searched Params for KNN", bestParams)
        file.write("Best Grid-Searched Params for KNN: "+str(bestParams)+'\n')
        file.flush()
        knnBestGridSearched = KNN.KNNLearner(**bestParams, datasetNo=set.datasetNo)
        exp.getLearningCurve(knnBestGridSearched, set)
        exp.getComplexityCurve(knnBestGridSearched, set)

        # repeat the learning process
        for i in range(steps):
            knnBestGridSearched = exp.repeatTheLearningProcess(knnBestGridSearched, set)

        score = exp.scoreTestingSet(knnBestGridSearched, set)

        print("Testing Score for KNN: ", score)
        file.write("Testing Score for KNN: " + str(score) + '\n')
        file.flush()

        # start the second learner Decision Tree
        dt = DT.DTLearner(datasetNo=set.datasetNo)
        _, bestParams = exp.getBestGridSearchedModel(dt, set)
        print("Best Grid-Searched Params for Decision Tree", bestParams)
        file.write("Best Grid-Searched Params for Decision Tree: " + str(bestParams)+'\n')
        file.flush()
        dtBestGridSearched = DT.DTLearner(**bestParams, datasetNo=set.datasetNo)
        exp.getLearningCurve(dtBestGridSearched, set)
        exp.getComplexityCurve(dtBestGridSearched, set)

        # repeat the learning process
        for i in range(steps):
            dtBestGridSearched = exp.repeatTheLearningProcess(dtBestGridSearched, set)

        score = exp.scoreTestingSet(dtBestGridSearched, set)

        print("Testing Score for Decision Tree: ", score)
        file.write("Testing Score for Decision Tree: " + str(score) + '\n')
        file.flush()

        # start the fourth learner SVM
        svm = SVM.SVMLearner(datasetNo=set.datasetNo)
        _, bestParams = exp.getBestGridSearchedModel(svm, set)
        print("Best Grid-Searched Params for SVM", bestParams)
        file.write("Best Grid-Searched Params for SVM: " + str(bestParams)+'\n')
        file.flush()
        svmBestGridSearched = SVM.SVMLearner(**bestParams, datasetNo=set.datasetNo)
        exp.getLearningCurve(svmBestGridSearched, set)
        exp.getComplexityCurve(svmBestGridSearched, set)

        # repeat the learning process
        for i in range(steps):
            svmBestGridSearched = exp.repeatTheLearningProcess(svmBestGridSearched, set)

        score = exp.scoreTestingSet(svmBestGridSearched, set)

        print("Testing Score for SVM: ", score)
        file.write("Testing Score for SVM: " + str(score) + '\n')
        file.flush()

        # start the fifth learner Boosting
        boosting = Boosting.BoostingLearner(datasetNo=set.datasetNo)
        _, bestParams = exp.getBestGridSearchedModel(boosting, set)
        print("Best Grid-Searched Params for Boosting", bestParams)
        file.write("Best Grid-Searched Params for Boosting: " + str(bestParams)+'\n')
        file.flush()
        boostingBestGridSearched = Boosting.BoostingLearner(**bestParams, datasetNo=set.datasetNo)
        exp.getLearningCurve(boostingBestGridSearched, set)
        exp.getComplexityCurve(boostingBestGridSearched, set)

        # repeat the learning process
        for i in range(steps):
            boostingBestGridSearched = exp.repeatTheLearningProcess(boostingBestGridSearched, set)

        score = exp.scoreTestingSet(boostingBestGridSearched, set)

        print("Testing Score for Boosting: ", score)
        file.write("Testing Score for Boosting: " + str(score) + '\n')
        file.flush()

        # start the third learner ANN
        ann = ANN.ANNLearner(datasetNo=set.datasetNo)
        _, bestParams = exp.getBestGridSearchedModel(ann, set)
        print("Best Grid-Searched Params for Neural Networks", bestParams)
        file.write("Best Grid-Searched Params for Neural Networks: " + str(bestParams)+'\n')
        file.flush()
        annBestGridSearched = ANN.ANNLearner(**bestParams, datasetNo=set.datasetNo)
        exp.getLearningCurve(annBestGridSearched, set)
        exp.getComplexityCurve(annBestGridSearched, set)
        exp.getTrainingTimeCurve(annBestGridSearched, set)

        # repeat the learning process
        for i in range(steps):
            annBestGridSearched = exp.repeatTheLearningProcess(annBestGridSearched, set)

        score = exp.scoreTestingSet(annBestGridSearched, set)

        print("Testing Score for Neural Network: ", score)
        file.write("Testing Score for Neural Network: " + str(score) + '\n')
        file.flush()

    file.close()
