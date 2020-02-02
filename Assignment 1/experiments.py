"""
Course:         CS 7641 Assignment 1, Spring 2020
Date:           January 19th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import KNN
import ANN
import DT
import SVM
import Boosting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.metrics import f1_score, accuracy_score


accuracy_scorer = 'accuracy'
f1_scorer = 'f1_weighted'

class experiments():
    def scoreTestingSet(self, BestGridSearched, set):
        BestGridSearched.learner.fit(set.training_x, set.training_y)

        if set.balanced == 1:
            score = accuracy_score(BestGridSearched.learner.predict(set.testing_x), set.testing_y)
        else:
            score = f1_score(BestGridSearched.learner.predict(set.testing_x), set.testing_y, average='weighted')

        return score

    def repeatTheLearningProcess(self, bestGridSearched, set):
        _, bestParams = self.getBestGridSearchedModel(bestGridSearched, set)

        if bestGridSearched.learnerType == 'KNN':
            bestGridSearched = KNN.KNNLearner(**bestParams, datasetNo=set.datasetNo)
        elif bestGridSearched.learnerType == 'DT':
            bestGridSearched = DT.DTLearner(**bestParams, datasetNo=set.datasetNo)
        elif bestGridSearched.learnerType == 'SVM':
            bestGridSearched = SVM.SVMLearner(**bestParams, datasetNo=set.datasetNo)
        elif bestGridSearched.learnerType == 'Boosting':
            bestGridSearched = Boosting.BoostingLearner(**bestParams, datasetNo=set.datasetNo)
        elif bestGridSearched.learnerType == 'ANN':
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
        if learnerType == 'KNN':
            n_samples = learner.n_neighbors

        scorer = accuracy_scorer
        if not isBalanced:
            scorer = f1_scorer

        if learnerType != 'ANN':
            train_sizes_m=[n_samples, int(len(X[:, 1])*0.2), int(len(X[:, 1])*0.4), int(len(X[:, 1])*0.6), int(len(X[:, 1])*0.8)]
            train_size, train_score, validation_score = learning_curve(
                estimator=learner,
                X=X,
                y=y,
                cv=5,
                train_sizes=train_sizes_m,
                scoring=scorer)
        else:
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
        plt.style.use('seaborn-whitegrid')
        plt.plot(train_size, train_scores_mean, label='Training', color='r', marker='o')
        plt.fill_between(train_size, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.25, color='r')
        plt.plot(train_size, validation_scores_mean, label='Validation', color='g', marker='o')
        plt.fill_between(train_size, validation_scores_mean - validation_scores_std, validation_scores_mean +
                         validation_scores_std, alpha=0.25, color='g')
        plt.ylabel('Score', fontsize=12)
        if dataset.datasetNo == 1:
            plt.ylim(0.5, 1.05)
        elif dataset.datasetNo == 2:
            plt.ylim(0.0, 1.05)
        plt.xlabel('Training set size', fontsize=12)
        plt.title('Learning curves for '+learnerType+' with Best Grid-Searched Parameters', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/'+learnerType+'-Learning-Curve, Dataset '+str(dataset.datasetNo)+'.png')
        plt.close()

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
        if learnerType != 'ANN':
            plt.plot(complexityParams[complexParamName], train_scores_mean, label='Training', color='r', marker='o')
            plt.fill_between(complexityParams[complexParamName], train_scores_mean - train_scores_std, train_scores_mean
                             + train_scores_std, alpha=0.25, color='r')
            plt.plot(complexityParams[complexParamName], validation_scores_mean, label='Validation', color='g', marker='o')
            plt.fill_between(complexityParams[complexParamName], validation_scores_mean - validation_scores_std,
                             validation_scores_mean + validation_scores_std, alpha=0.25, color='g')
            plt.plot(complexityParams[complexParamName][np.argmax(validation_scores_mean)],
                     np.max(validation_scores_mean), color='k', marker='o')

            bestValuelist = []
            bestValue = complexityParams[complexParamName][np.argmax(validation_scores_mean)]
            bestValuelist.append(bestValue)
            learnerClass.setGridSearchParams(complexParamName, bestValuelist)

        else:
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
            bestValuelist.append(bestValue)
            learnerClass.setGridSearchParams(complexParamName, bestValuelist)

        plt.ylim(0.5, 1.05)
        plt.ylabel('Score', fontsize=12)
        plt.xlabel(complexParamName, fontsize=12)
        plt.title('Complexity curve for ' + learnerType, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/' + learnerType + '-Complexity-Curve, Dataset '+str(dataset.datasetNo)+'.png')
        plt.close()

        if learnerType == 'SVM':
            complexParamName = list(complexityParams.keys())[1]

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
            plt.plot(np.log10(complexityParams[complexParamName]), train_scores_mean, label='Training', color='r', marker='o')
            plt.fill_between(np.log10(complexityParams[complexParamName]), train_scores_mean - train_scores_std,
                             train_scores_mean
                             + train_scores_std, alpha=0.25, color='r')
            plt.plot(np.log10(complexityParams[complexParamName]), validation_scores_mean, label='Validation', color='g',
                     marker='o')
            plt.fill_between(np.log10(complexityParams[complexParamName]), validation_scores_mean - validation_scores_std,
                             validation_scores_mean + validation_scores_std, alpha=0.25, color='g')
            plt.plot(np.log10(complexityParams[complexParamName][np.argmax(validation_scores_mean)]),
                     np.max(validation_scores_mean), color='k', marker='o')

            bestValuelist = []
            bestValue = complexityParams[complexParamName][np.argmax(validation_scores_mean)]
            bestValuelist.append(bestValue)
            learnerClass.setGridSearchParams(complexParamName, bestValuelist)

            if dataset.datasetNo == 1:
                plt.ylim(0.5, 1.05)
            elif dataset.datasetNo == 2:
                plt.ylim(0.0, 1.05)
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('log('+complexParamName+')', fontsize=12)
            plt.title('Complexity curve for ' + learnerType, fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + learnerType + '-Complexity-Curve, 2, Dataset ' + str(dataset.datasetNo) + '.png')
            plt.close()

        return bestValue

    def getTrainingTimeCurve(self, learnerClass, dataset):
        learnerType = learnerClass.getLearnerType()
        X = dataset.training_x
        y = dataset.training_y
        isBalanced = dataset.balanced

        if learnerType == 'ANN':
            learner = learnerClass.getLearner()
            trainingTimeParam = learnerClass.getTrainingTimeParam()

            scorer = accuracy_scorer
            if not isBalanced:
                scorer = f1_scorer

            trainingTimeName = 'max_iter'

            train_scores, test_scores = validation_curve(learner,
                                                         X,
                                                         y,
                                                         trainingTimeName,
                                                         trainingTimeParam[trainingTimeName],
                                                         cv=5,
                                                         scoring=scorer)

            train_scores_mean = train_scores.mean(axis=1)
            train_scores_std = train_scores.std(axis=1)
            validation_scores_mean = test_scores.mean(axis=1)
            validation_scores_std = test_scores.std(axis=1)
            plt.style.use('seaborn-whitegrid')
            plt.plot(trainingTimeParam[trainingTimeName], train_scores_mean, label='Training', color='r', marker='o')
            plt.fill_between(trainingTimeParam[trainingTimeName], train_scores_mean - train_scores_std, train_scores_mean
                             + train_scores_std, alpha=0.25, color='r')
            plt.plot(trainingTimeParam[trainingTimeName], validation_scores_mean, label='Validation', color='g', marker='o')
            plt.fill_between(trainingTimeParam[trainingTimeName], validation_scores_mean - validation_scores_std,
                             validation_scores_mean + validation_scores_std, alpha=0.25, color='g')
            plt.plot(trainingTimeParam[trainingTimeName][np.argmax(validation_scores_mean)],
                     np.max(validation_scores_mean), color='k', marker='o')
            bestValue = trainingTimeParam[trainingTimeName][np.argmax(validation_scores_mean)]

            plt.ylim(0.5, 1.05)
            plt.ylabel('Score', fontsize=12)
            plt.xlabel(trainingTimeName, fontsize=12)
            plt.title('Performance curve for ' + learnerType, fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + learnerType + '-Training-Time-Curve, Dataset '+str(dataset.datasetNo)+'.png')
            plt.close()

            return bestValue

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