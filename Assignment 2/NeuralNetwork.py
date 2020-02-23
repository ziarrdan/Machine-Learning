"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import sklearn.model_selection as ms
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import dataloader
import numpy as np
import mlrose
import time


class neuralNetworkWeightOptimization:
    def __init__(self):
        self.dataset1 = dataloader.wineDS()
        self.dataset1.processDataset()
        self.start_time = 0.
        self.times = []
        self.optimizers = ['gradient_descent', 'simulated_annealing', 'genetic_algorithm', 'random_hill_climb']
        self.NNmaxIter = []
    def optimizeNeuralNetworWeights(self, optimizer):
        self.NNmaxIter = np.arange(5, 1006, 100)
        if (optimizer == 'genetic_alg'):
            self.NNmaxIter = np.arange(5, 1006, 250)
        kFold = 5
        kFoldGen = 1
        iteration = 0
        learner = []
        legend = []
        timeTraining = []
        learnersTrainingScore = []
        learnersTestingScore = []
        learnersTrainingScoreStd = []
        learnersTestingScoreStd = []
        kFoldTrainingScore = []
        kFoldTestingScore = []
        learnersTime = []
        learnersTimeStd = []
        kFoldTime = []

        for iter in self.NNmaxIter:
            learner.append([])
            legend.append([])
            if (optimizer == 'gradient_descent'):
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='gradient_descent',
                                            max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                            early_stopping=False, clip_max=1e10, curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('GA')
            elif (optimizer == 'simulated_annealing'):
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                            max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                            early_stopping=False, clip_max=1e10, schedule=mlrose.ExpDecay(),
                                            curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('Exp')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, schedule=mlrose.ArithDecay(),
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('Arithmetic')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, schedule=mlrose.GeomDecay(),
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('Geometric')
            elif (optimizer == 'genetic_alg'):
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                            max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                            early_stopping=False, clip_max=1e10, pop_size=300, mutation_prob=0.2,
                                            curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('300')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, pop_size=200, mutation_prob=0.2,
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('200')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, pop_size=100, mutation_prob=0.2,
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('100')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, pop_size=200, mutation_prob=0.1,
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('0.1')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, pop_size=200, mutation_prob=0.2,
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('0.2')
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                          max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                          early_stopping=False, clip_max=1e10, pop_size=200, mutation_prob=0.3,
                                          curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('0.3')
                kFold = kFoldGen
            elif (optimizer == 'random_hill_climb'):
                nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                             max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                             early_stopping=False, clip_max=1e10, curve=False)
                learner[iteration].append(nn)
                legend[iteration].append('RHC')

            iteration += 1

        for l in range(len(learner[0])):
            learnersTrainingScore.append([])
            learnersTestingScore.append([])
            learnersTrainingScoreStd.append([])
            learnersTestingScoreStd.append([])
            learnersTime.append([])
            learnersTimeStd.append([])
            for iter in range(len(self.NNmaxIter)):
                learnersTrainingScore[l].append(l)
                learnersTestingScore[l].append(l)
                learnersTrainingScoreStd[l].append(l)
                learnersTestingScoreStd[l].append(l)
                learnersTime[l].append(l)
                learnersTimeStd[l].append(l)
                kFoldTrainingScore = []
                kFoldTestingScore = []
                kFoldTime = []
                for k in range(kFold):
                    training_x, testing_x, training_y, testing_y = ms.train_test_split(
                        self.dataset1.training_x, self.dataset1.training_y, test_size=0.2, random_state=self.dataset1.seed,
                        stratify=self.dataset1.training_y)

                    timeStart = time.time()
                    learner[iter][l].fit(training_x, training_y)
                    timeTraining = (time.time() - timeStart)

                    trainingF1 = f1_score(learner[iter][l].predict(training_x), training_y, average='weighted')
                    testingF1 = f1_score(learner[iter][l].predict(testing_x), testing_y, average='weighted')
                    kFoldTrainingScore.append(trainingF1)
                    kFoldTestingScore.append(testingF1)
                    kFoldTime.append(timeTraining)

                learnersTrainingScore[l][iter] = (np.mean(kFoldTrainingScore))
                learnersTestingScore[l][iter] = (np.mean(kFoldTestingScore))
                learnersTrainingScoreStd[l][iter] = (np.std(kFoldTrainingScore))
                learnersTestingScoreStd[l][iter] = (np.std(kFoldTestingScore))
                learnersTime[l][iter] = (np.mean(kFoldTime))
                learnersTimeStd[l][iter] = (np.std(kFoldTime))

        learnersTrainingScore = np.array(learnersTrainingScore)
        learnersTestingScore = np.array(learnersTestingScore)
        learnersTrainingScoreStd = np.array(learnersTrainingScoreStd)
        learnersTestingScoreStd = np.array(learnersTestingScoreStd)
        learnersTime = np.array(learnersTime)
        learnersTimeStd = np.array(learnersTimeStd)

        learnerCount = 0
        if (optimizer != 'genetic_alg'):
            plt.style.use('seaborn-whitegrid')
            for l in range(len(learnersTrainingScore)):
                plt.plot(self.NNmaxIter, learnersTrainingScore[l], label='Train for ' + legend[0][l])
                plt.plot(self.NNmaxIter, learnersTestingScore[l], label='Valid. for ' + legend[0][l])
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Number of Iterations', fontsize=12)
            plt.title('F1 Score for NN trained using ' + optimizer, fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + optimizer + '-F1-Score.png')
            plt.close()
        else:
            plt.style.use('seaborn-whitegrid')
            splitParams = int(len(learnersTrainingScore) / 2)
            for l in range(splitParams):
                plt.plot(self.NNmaxIter, learnersTrainingScore[l], label='Train for ' + legend[0][l])
                plt.plot(self.NNmaxIter, learnersTestingScore[l], label='Valid. for ' + legend[0][l])
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Number of Iterations', fontsize=12)
            plt.title('F1 Score for NN trained using ' + optimizer, fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + optimizer + '-F1-Score, 1.png')
            plt.close()

            plt.style.use('seaborn-whitegrid')
            for l in range(splitParams):
                plt.plot(self.NNmaxIter, learnersTrainingScore[l+splitParams], label='Train for ' + legend[0][l+splitParams])
                plt.plot(self.NNmaxIter, learnersTestingScore[l+splitParams], label='Valid. for ' + legend[0][l+splitParams])
            plt.ylabel('Score', fontsize=12)
            plt.xlabel('Number of Iterations', fontsize=12)
            plt.title('F1 Score for NN trained using ' + optimizer, fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + optimizer + '-F1-Score, 2.png')
            plt.close()

        return learnersTrainingScore, learnersTrainingScoreStd, learnersTestingScore, learnersTestingScoreStd, learnersTime, learnersTimeStd

    def optimizeNNs(self):
        self.optimizeNeuralNetworWeights('gradient_descent')
        self.optimizeNeuralNetworWeights('simulated_annealing')
        self.optimizeNeuralNetworWeights('genetic_alg')
        self.optimizeNeuralNetworWeights('random_hill_climb')

    def compareNNs(self):
        self.NNmaxIter = np.arange(5, 1006, 100)
        self.NNmaxIterGen = np.arange(5, 1006, 250)
        kFold = 5
        kFoldGen = 1
        iteration = 0
        learner = []
        legend = []
        timeTraining = []
        learnersTrainingScore = []
        learnersTestingScore = []
        learnersTrainingScoreStd = []
        learnersTestingScoreStd = []
        learnersTime = []
        learnersTimeStd = []

        for iter in self.NNmaxIter:
            learner.append([])
            legend.append([])

            nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='gradient_descent',
                                      max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                      early_stopping=False, clip_max=1e10, curve=False)
            learner[iteration].append(nn)
            legend[iteration].append('GD')
            nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                      max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                      early_stopping=False, clip_max=1e10, schedule=mlrose.GeomDecay(),
                                      curve=False)
            learner[iteration].append(nn)
            legend[iteration].append('SA')
            nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                      max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                      early_stopping=False, clip_max=1e10, curve=False)
            learner[iteration].append(nn)
            legend[iteration].append('RHC')
            nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                      max_iters=int(iter), bias=True, is_classifier=True, learning_rate=0.001,
                                      early_stopping=False, clip_max=1e10, pop_size=300, mutation_prob=0.2,
                                      curve=False)
            learner[iteration].append(nn)
            legend[iteration].append('GA')

            iteration += 1

        for l in range(len(learner[0])):
            learnersTrainingScore.append([])
            learnersTestingScore.append([])
            learnersTrainingScoreStd.append([])
            learnersTestingScoreStd.append([])
            learnersTime.append([])
            learnersTimeStd.append([])
            if legend[0][l] == 'GA':
                NNmaxIter = self.NNmaxIterGen
                kFold = 5
            else:
                NNmaxIter = self.NNmaxIter
                kFold = 1
            for iter in range(len(NNmaxIter)):
                learnersTrainingScore[l].append(l)
                learnersTestingScore[l].append(l)
                learnersTrainingScoreStd[l].append(l)
                learnersTestingScoreStd[l].append(l)
                learnersTime[l].append(l)
                learnersTimeStd[l].append(l)
                kFoldTrainingScore = []
                kFoldTestingScore = []
                kFoldTime = []
                for k in range(kFold):
                    training_x, testing_x, training_y, testing_y = ms.train_test_split(
                        self.dataset1.training_x, self.dataset1.training_y, test_size=0.2,
                        random_state=self.dataset1.seed,
                        stratify=self.dataset1.training_y)

                    timeStart = time.time()
                    learner[iter][l].fit(training_x, training_y)
                    timeTraining = (time.time() - timeStart)

                    trainingF1 = f1_score(learner[iter][l].predict(training_x), training_y, average='weighted')
                    testingF1 = f1_score(learner[iter][l].predict(testing_x), testing_y, average='weighted')
                    kFoldTrainingScore.append(trainingF1)
                    kFoldTestingScore.append(testingF1)
                    kFoldTime.append(timeTraining)

                learnersTrainingScore[l][iter] = (np.mean(kFoldTrainingScore))
                learnersTestingScore[l][iter] = (np.mean(kFoldTestingScore))
                learnersTrainingScoreStd[l][iter] = (np.std(kFoldTrainingScore))
                learnersTestingScoreStd[l][iter] = (np.std(kFoldTestingScore))
                learnersTime[l][iter] = (np.mean(kFoldTime))
                learnersTimeStd[l][iter] = (np.std(kFoldTime))

        learnersTrainingScore = np.array(learnersTrainingScore)
        learnersTime = np.array(learnersTime)

        plt.style.use('seaborn-whitegrid')
        for l in range(len(learnersTrainingScore)):
            if legend[0][l] == 'GA':
                NNmaxIter = self.NNmaxIterGen
                kFold = 5
            else:
                NNmaxIter = self.NNmaxIter
                kFold = 1
            plt.plot(NNmaxIter, learnersTrainingScore[l], label=legend[0][l])
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Number of Iterations', fontsize=12)
        plt.title('Comparing Validation F1 Score for All Optimizers', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Comparing-F1-Score-for-All.png')
        plt.close()

        plt.style.use('seaborn-whitegrid')
        for l in range(len(learnersTrainingScore)):
            if legend[0][l] == 'GA':
                NNmaxIter = self.NNmaxIterGen
                kFold = 5
            else:
                NNmaxIter = self.NNmaxIter
                kFold = 1
            plt.plot(NNmaxIter, learnersTime[l], label=legend[0][l])
        plt.ylabel('Time (s)', fontsize=12)
        plt.xlabel('Number of Iterations', fontsize=12)
        plt.title('Comparing Training Time for All Optimizers', fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Comparing-Time-for-All.png')
        plt.close()

    def calcTestScore(self):
        legendCounter = 0
        learners = []
        legend = []

        nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='gradient_descent',
                                  max_iters=200, bias=True, is_classifier=True, learning_rate=0.001,
                                  early_stopping=False, clip_max=1e10, curve=False)
        learners.append(nn)
        legend.append('GD')
        nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='simulated_annealing',
                                  max_iters=800, bias=True, is_classifier=True, learning_rate=0.001,
                                  early_stopping=False, clip_max=1e10, schedule=mlrose.GeomDecay(),
                                  curve=False)
        learners.append(nn)
        legend.append('SA')
        nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='random_hill_climb',
                                  max_iters=600, bias=True, is_classifier=True, learning_rate=0.001,
                                  early_stopping=False, clip_max=1e10, curve=False)
        learners.append(nn)
        legend.append('RHC')
        nn = mlrose.NeuralNetwork(hidden_nodes=[20], activation='relu', algorithm='genetic_alg',
                                  max_iters=200, bias=True, is_classifier=True, learning_rate=0.001,
                                  early_stopping=False, clip_max=1e10, pop_size=300, mutation_prob=0.2,
                                  curve=False)
        learners.append(nn)
        legend.append('GA')

        for learner in learners:
            timeStart = time.time()
            learner.fit(self.dataset1.training_x, self.dataset1.training_y)
            testingF1 = f1_score(learner.predict(self.dataset1.testing_x), self.dataset1.testing_y, average='weighted')
            print('Testing F1-Score for' + legend[legendCounter] + ': ' + str(testingF1))
            legendCounter += 1
