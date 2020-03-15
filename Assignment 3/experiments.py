"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import ANN
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import f1_score


accuracy_scorer = 'accuracy'
f1_scorer = 'f1_weighted'

class experiments():
    def scoreTestingSet(self, BestGridSearched, set):
        BestGridSearched.learner.fit(set.training_x, set.training_y)

        score = f1_score(BestGridSearched.learner.predict(set.testing_x), set.testing_y, average='weighted')

        return score

    def repeatTheLearningProcess(self, bestGridSearched, set):
        _, bestParams = self.getBestGridSearchedModel(bestGridSearched, set)

        if bestGridSearched.learnerType == 'ANN':
            if 'Income' in set.name:
                set.datasetNo = 1
            elif 'Wine' in set.name:
                set.datasetNo = 2
            bestGridSearched = ANN.ANNLearner(**bestParams, datasetNo=set.datasetNo)

        self.getLearningCurve(bestGridSearched, set)
        self.getComplexityCurve(bestGridSearched, set)

        return bestGridSearched

    def getLearningCurve(self, learnerClass, dataset):
        learner = learnerClass.getLearner()
        learnerType = learnerClass.getLearnerType()
        X = dataset.training_x
        y = dataset.training_y
        isBalanced = dataset.balanced

        n_samples = 1

        scorer = accuracy_scorer
        if not isBalanced:
            scorer = f1_scorer

        train_sizes_m = [n_samples, int(len(X[:, 1]) * 0.2), int(len(X[:, 1]) * 0.4), int(len(X[:, 1]) * 0.6),
                         int(len(X[:, 1]) * 0.8)]
        train_size, train_score, validation_score, fit_times, score_times = learning_curve(
            estimator=learner,
            X=X,
            y=y,
            cv=5,
            train_sizes=train_sizes_m,
            scoring=scorer,
            return_times = True)

        train_scores_mean = train_score.mean(axis=1)
        validation_scores_mean = validation_score.mean(axis=1)
        train_scores_std= train_score.std(axis=1)
        validation_scores_std = validation_score.std(axis=1)

        return train_size, train_scores_mean, train_scores_std, validation_scores_mean, validation_scores_std

    def getComplexityCurve(self, learnerClass, dataset):
        learner = learnerClass.getLearner()
        learnerType = learnerClass.getLearnerType()
        complexityParams = learnerClass.getComplexityParams()
        X = dataset.training_x
        y = dataset.training_y
        isBalanced = dataset.balanced

        scorer = accuracy_scorer
        if not isBalanced:
            scorer = f1_scorer

        complexParamName =list(complexityParams.keys())[0]

        train_scores, test_scores = validation_curve(learner,
                                                     X,
                                                     y,
                                                     complexParamName,
                                                     complexityParams[complexParamName],
                                                     cv=5,
                                                     scoring=scorer)

        train_scores_mean = train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        validation_scores_mean = test_scores.mean(axis=1)
        validation_scores_std = test_scores.std(axis=1)
        plt.style.use('seaborn-whitegrid')

        xRange = [i[0] for i in complexityParams[complexParamName]]
        plt.plot(xRange, train_scores_mean, label='Training', color='r', marker='o')
        plt.fill_between(xRange, train_scores_mean - train_scores_std, train_scores_mean
                         + train_scores_std, alpha=0.25, color='r')

        plt.plot(xRange, validation_scores_mean, label='Validation', color='g', marker='o')
        plt.fill_between(xRange, validation_scores_mean - validation_scores_std,
                         validation_scores_mean + validation_scores_std, alpha=0.25, color='g')

        plt.plot(xRange[np.argmax(validation_scores_mean)], np.max(validation_scores_mean), color='k', marker='o')

        bestValuelist = []
        bestValue = complexityParams[complexParamName][np.argmax(validation_scores_mean)]
        print("The optimum value found using complexity analysis is: ", str(bestValue))
        promptText = "If you do not want the optimum value for "+ complexParamName +" to be used at the next stage " \
                     "for "+ learnerType +" learner, please type yours here: "
        bestManuelValue = input(promptText)
        if bestManuelValue != "":
            bestValuelist.append((int(bestManuelValue.split(',')[0][1:]),))
        else:
            bestValuelist.append(bestValue)
        learnerClass.setGridSearchParams(complexParamName, bestValuelist)

        if 'Income' in dataset.name:
            plt.ylim(0.5, 1.05)
        elif 'Wine' in dataset.name:
            plt.ylim(0.0, 1.05)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel(complexParamName, fontsize=12)
        plt.title('Complexity curve for ' + learnerType, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/ANN/' + learnerType + '-Complexity-Curve, Dataset '+str(dataset.name)+'.png')
        plt.close()

        return bestValue

    def getTrainingTimeCurve(self, learnerClass, dataset):
        X = dataset.training_x
        Y = dataset.training_y
        learner = learnerClass.getLearner()

        timeArr = []
        times = range(10, len(X), int((len(X) - 10) / 10))
        timeTotal = 0

        for length in times:
            for i in range(10):
                start = time.time()
                learner.fit(X[:length], Y[:length])
                stop = time.time()
                timeTotal += stop - start
            timeTotal /= 10
            timeArr.append(timeTotal)

        return times, timeArr

    def getBestGridSearchedModel(self, learnerClass, dataset):
        learner = learnerClass.getLearner()
        gridSearchParams = learnerClass.getGridSearchParams()
        X = dataset.training_x
        y = dataset.training_y
        isBalanced = dataset.balanced

        scorer = accuracy_scorer
        if not isBalanced:
            scorer = f1_scorer

        bestGridSearchedModel = GridSearchCV(
            learner,
            param_grid=gridSearchParams,
            refit=True,
            cv=5,
            scoring=scorer)

        bestGridSearchedModel.fit(X, y)

        return bestGridSearchedModel.best_estimator_, bestGridSearchedModel.best_params_

    def getLearningCurveAll(self, Datasets, isWithClusters=False):
        color = ['red', 'blue', 'green', 'orange', 'purple']
        cntr = 0
        for set in Datasets:
            # start the third learner ANN
            if 'Income' in set.name:
                set.datasetNo = 1
                if isWithClusters == False:
                    thisSetName = 'Income'
                else:
                    thisSetName = 'Income, 2'
                annBestGridSearched = ANN.ANNLearner(datasetNo=set.datasetNo, hidden_layer_sizes=(11,))
            elif 'Wine' in set.name:
                set.datasetNo = 2
                if isWithClusters == False:
                    thisSetName = 'Wine'
                else:
                    thisSetName = 'Wine, 2'
                annBestGridSearched = ANN.ANNLearner(datasetNo=set.datasetNo, hidden_layer_sizes=(15,))

            x, train_mean, train_std, test_mean, test_std = self.getLearningCurve(annBestGridSearched, set)

            score = self.scoreTestingSet(annBestGridSearched, set)

            plt.plot(x, train_mean, label=set.name, marker='o', color=color[cntr % 5])
            plt.fill_between(x, train_mean - train_std, train_mean + train_std,
                             alpha=0.25)
            plt.plot(x, test_mean, marker='o', color=color[cntr % 5])
            plt.fill_between(x, test_mean - test_std, test_mean +
                             test_std, alpha=0.25)
            cntr += 1

        plt.style.use('seaborn-whitegrid')
        plt.ylabel('Score', fontsize=12)
        if thisSetName == 'Income':
            plt.ylim(0.5, 1.05)
        else:
            plt.ylim(0.0, 1.05)
        plt.xlabel('Training set size', fontsize=12)
        plt.title('Learning curves for all ' + thisSetName.split(',')[0] + ' Datasets', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/ANN/All-Learning-Curve, Dataset ' + thisSetName + '.png')
        plt.close()

        cntr = 0
        for set in Datasets:
            # start the third learner ANN
            if 'Income' in set.name:
                set.datasetNo = 1
            elif 'Wine' in set.name:
                set.datasetNo = 2

            annBestGridSearched = ANN.ANNLearner(datasetNo=set.datasetNo)
            time, train_time = self.getTrainingTimeCurve(annBestGridSearched, set)

            score = self.scoreTestingSet(annBestGridSearched, set)
            print("Testing Score for " + set.name + " is: ", score)

            plt.plot(time, train_time, label=set.name, marker='o', color=color[cntr % 5])
            cntr += 1

        plt.style.use('seaborn-whitegrid')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Number of Training Samples', fontsize=12)
        plt.title('Training Time for all ' + thisSetName.split(',')[0] + ' Datasets', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/ANN/All-Training-Time-Curve, Dataset ' + thisSetName + '.png')
        plt.close()