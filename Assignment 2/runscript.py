"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import mlrose
import numpy as np
import RandomizedHillClimbing as rhc
import SimulatedAnnealing as sa
import GeneticAlgorithm as ga
import MIMIC as mimic
import experiments
import NeuralNetwork
import random

if __name__ == '__main__':
    optimizers = []
    exp = experiments.experiments()

    """"# First problem is the Knapsack
    weights = np.random.randint(1, 20, size=20)
    values = np.random.randint(1, 10, size=20)
    max_weight_pct = 0.65
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    # Define optimization problem object
    problemFit = mlrose.DiscreteOpt(length=20, fitness_fn=fitness, maximize=True)
    # Create and run the Randomized Hill Climbing Optimizer class
    rhcOptimizer = rhc.RHC()
    optimizers.append(rhcOptimizer)
    exp.getComplexityCurve(optimizer=rhcOptimizer, problem=problemFit, problemName='Knapsack')
    # Create and run the Simulated Annealing Optimizer class
    saOptimizer = sa.SA()
    optimizers.append(saOptimizer)
    exp.getComplexityCurve(optimizer=saOptimizer, problem=problemFit, problemName='Knapsack')
    # Create and run the Genetic Algorithm Optimizer class
    gaOptimizer = ga.GA()
    optimizers.append(gaOptimizer)
    exp.getComplexityCurve(optimizer=gaOptimizer, problem=problemFit, problemName='Knapsack')
    # Create and run the MIMIC Optimizer class
    mimicOptimizer = mimic.MIMIC()
    optimizers.append(mimicOptimizer)
    exp.getComplexityCurve(optimizer=mimicOptimizer, problem=problemFit, problemName='Knapsack')
    exp.getComparisonCurve(optimizers, problem=problemFit, problemName='Knapsack')

    # Second problem is the One Max
    optimizers = []
    fitness = mlrose.OneMax()
    # Define optimization problem object
    problemFit = mlrose.DiscreteOpt(length=20, fitness_fn=fitness, maximize=True, max_val=2)
    # Create and run the Randomized Hill Climbing Optimizer class
    rhcOptimizer = rhc.RHC()
    optimizers.append(rhcOptimizer)
    exp.getComplexityCurve(optimizer=rhcOptimizer, problem=problemFit, problemName='One Max')
    # Create and run the Simulated Annealing Optimizer class
    saOptimizer = sa.SA()
    optimizers.append(saOptimizer)
    exp.getComplexityCurve(optimizer=saOptimizer, problem=problemFit, problemName='One Max')
    # Create and run the Genetic Algorithm Optimizer class
    gaOptimizer = ga.GA()
    optimizers.append(gaOptimizer)
    exp.getComplexityCurve(optimizer=gaOptimizer, problem=problemFit, problemName='One Max')
    # Create and run the MIMIC Optimizer class
    mimicOptimizer = mimic.MIMIC()
    optimizers.append(mimicOptimizer)
    exp.getComplexityCurve(optimizer=mimicOptimizer, problem=problemFit, problemName='One Max')
    exp.getComparisonCurve(optimizers, problem=problemFit, problemName='One Max')

    # Third problem is the Four Peaks
    optimizers = []
    fitness = mlrose.FourPeaks(t_pct=0.1)
    # Define optimization problem object
    problemFit = mlrose.DiscreteOpt(length=20, fitness_fn=fitness, maximize=True, max_val=2)
    # Create and run the Randomized Hill Climbing Optimizer class
    rhcOptimizer = rhc.RHC()
    optimizers.append(rhcOptimizer)
    exp.getComplexityCurve(optimizer=rhcOptimizer, problem=problemFit, problemName='Four Peaks')
    # Create and run the Simulated Annealing Optimizer class
    saOptimizer = sa.SA()
    optimizers.append(saOptimizer)
    exp.getComplexityCurve(optimizer=saOptimizer, problem=problemFit, problemName='Four Peaks')
    # Create and run the Genetic Algorithm Optimizer class
    gaOptimizer = ga.GA()
    optimizers.append(gaOptimizer)
    exp.getComplexityCurve(optimizer=gaOptimizer, problem=problemFit, problemName='Four Peaks')
    # Create and run the MIMIC Optimizer class
    mimicOptimizer = mimic.MIMIC()
    optimizers.append(mimicOptimizer)
    exp.getComplexityCurve(optimizer=mimicOptimizer, problem=problemFit, problemName='Four Peaks')
    exp.getComparisonCurve(optimizers, problem=problemFit, problemName='Four Peaks')"""

    nn = NeuralNetwork.neuralNetworkWeightOptimization()
    nn.optimizeNNs()
    nn.compareNNs()
