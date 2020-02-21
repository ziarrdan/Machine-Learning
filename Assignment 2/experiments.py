"""
Course:         CS 7641 Assignment 2, Spring 2020
Date:           February 13th, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

import matplotlib.pyplot as plt
import numpy as np
import mlrose
import time

start_time = 0.
times = []


def time_callback(iteration, attempt=None, done=None, state=None, fitness=None, curve=None, user_data=None):
    """Time callback for saving time elapsed at each iteration of the algorithm.
        Args:
          iteration (int): current iteration.
          attempt (int): current attempt.
          done (bool): id we are done iterating.
          state (list): current best state.
          fitness (float): current best fitness.
          curve (ndarray): current fitness curve.
          user_data (any): current iteration.
        Returns:
          continue (bool): True, to continue iterating.
        """
    # Define global variables
    global start_time, times

    # At first iteration, save start time and reset list of times, else save time elapsed since start
    if iteration == 0:
        start_time = time.time()
        times = []
    else:
        times.append(time.time() - start_time)

    # Return always True to continue iterating
    return True


class experiments():
    def getComplexityCurve(self, optimizer, problem, problemName):
        numerOfTrials = 10
        numerOfTrialsForGAandMIMIC = 5
        Curve= []
        Time = []

        if optimizer.getOptimizerName() == 'Randomized Hill Climbing':
            for i in range(numerOfTrials):
                _, state, curve = mlrose.random_hill_climb(problem,
                                                           max_attempts=optimizer.max_attempts,
                                                           max_iters=int(max(optimizer.parameters['max_iters'])),
                                                           restarts=optimizer.restarts,
                                                           init_state=optimizer.init_state,
                                                           curve=True,
                                                           state_fitness_callback=time_callback,
                                                           random_state=optimizer.random_state,
                                                           callback_user_info=[])

                curve = np.array(curve)

                if len(curve) < int(max(optimizer.parameters['max_iters'])):
                    for i in range(int(int(max(optimizer.parameters['max_iters'])) - len(curve))):
                        curve = np.append(curve, curve[-1])

                Curve.append(curve)
                Time.append(times)

            rhcCurve_mean = np.array(Curve).mean(axis=0)
            rhcCurve_std = np.array(Curve).std(axis=0)
            complexParamName = 'max_iters'

            plt.style.use('seaborn-whitegrid')
            plt.plot(optimizer.parameters[complexParamName], rhcCurve_mean, label='Fitness')
            plt.fill_between(optimizer.parameters[complexParamName], rhcCurve_mean - rhcCurve_std,
                             rhcCurve_mean + rhcCurve_std, alpha=0.25)
            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, ' + problemName + '.png')
            plt.close()

        Curve = []
        Time = []

        if optimizer.getOptimizerName() == 'Simulated Annealing':
            for schedule in range(len(optimizer.parameters['schedule'])):
                for i in range(numerOfTrials):
                    _, state, curve = mlrose.simulated_annealing(problem,
                                                                 schedule=optimizer.parameters['schedule'][schedule],
                                                                 max_attempts=optimizer.max_attempts,
                                                                 max_iters=int(max(optimizer.parameters['max_iters'])),
                                                                 init_state=optimizer.init_state,
                                                                 curve=True,
                                                                 state_fitness_callback=time_callback,
                                                                 random_state=optimizer.random_state,
                                                                 callback_user_info=[])

                    curve = np.array(curve)

                    if len(curve) < int(max(optimizer.parameters['max_iters'])):
                        for i in range(int(int(max(optimizer.parameters['max_iters'])) - len(curve))):
                            curve = np.append(curve, curve[-1])

                    Curve.append(curve)
                    Time.append(times)

                saCurve_mean = np.array(Curve).mean(axis=0)
                saCurve_std = np.array(Curve).std(axis=0)
                complexParamName = 'max_iters'

                plt.style.use('seaborn-whitegrid')
                if schedule == 0:
                    plt.plot(optimizer.parameters[complexParamName], saCurve_mean, label='Geometric Decay')
                elif schedule == 1:
                    plt.plot(optimizer.parameters[complexParamName], saCurve_mean, label='Arithmetic Decay')
                elif schedule == 2:
                    plt.plot(optimizer.parameters[complexParamName], saCurve_mean, label='Exponential Decay')
                plt.fill_between(optimizer.parameters[complexParamName], saCurve_mean - saCurve_std,
                                 saCurve_mean + saCurve_std, alpha=0.25)

            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, ' + problemName + '.png')
            plt.close()

        Curve = []
        Time = []

        if optimizer.getOptimizerName() == 'Genetic Algorithm':
            for pop in range(len(optimizer.parameters['pop_size'])):
                for i in range(numerOfTrialsForGAandMIMIC):
                    _, state, curve = mlrose.genetic_alg(problem,
                                                         pop_size=int(optimizer.parameters['pop_size'][pop]),
                                                         mutation_prob=optimizer.mutation_prob,
                                                         max_attempts=optimizer.max_attempts,
                                                         max_iters=int(max(optimizer.parameters['max_iters'])),
                                                         curve=True,
                                                         state_fitness_callback=time_callback,
                                                         random_state = optimizer.random_state,
                                                         callback_user_info=[])

                    curve = np.array(curve)

                    if len(curve) < int(max(optimizer.parameters['max_iters'])):
                        for i in range(int(int(max(optimizer.parameters['max_iters']))-len(curve))):
                            curve = np.append(curve, curve[-1])

                    Curve.append(curve)
                    Time.append(times)

                gaCurve_mean = np.array(Curve).mean(axis=0)
                gaCurve_std = np.array(Curve).std(axis=0)
                complexParamName = 'max_iters'

                plt.style.use('seaborn-whitegrid')
                plt.plot(optimizer.parameters[complexParamName], gaCurve_mean, label=str(optimizer.parameters['pop_size'][pop]))
                plt.fill_between(optimizer.parameters[complexParamName], gaCurve_mean - gaCurve_std,
                                 gaCurve_mean + gaCurve_std, alpha=0.25)

            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration w.r.t. Population Size for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            if problemName == 'One Max' or problemName == 'Continuous Peaks':
                plt.xlim(0, 16)
            elif problemName == 'Knapsack':
                plt.xlim(0, 20)
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, 1 ' + problemName + '.png')
            plt.close()

            Curve = []
            Time = []

            for mut in range(len(optimizer.parameters['mutation_prob'])):
                for i in range(numerOfTrialsForGAandMIMIC):
                    _, state, curve = mlrose.genetic_alg(problem,
                                                         pop_size=int(optimizer.pop_size),
                                                         mutation_prob=optimizer.parameters['mutation_prob'][mut],
                                                         max_attempts=optimizer.max_attempts,
                                                         max_iters=int(max(optimizer.parameters['max_iters'])),
                                                         curve=True,
                                                         state_fitness_callback=time_callback,
                                                         random_state = optimizer.random_state,
                                                         callback_user_info=[])

                    curve = np.array(curve)

                    if len(curve) < int(max(optimizer.parameters['max_iters'])):
                        for i in range(int(int(max(optimizer.parameters['max_iters'])) - len(curve))):
                            curve = np.append(curve, curve[-1])

                    Curve.append(curve)
                    Time.append(times)

                gaCurve_mean = np.array(Curve).mean(axis=0)
                gaCurve_std = np.array(Curve).std(axis=0)
                complexParamName = 'max_iters'

                plt.style.use('seaborn-whitegrid')
                plt.plot(optimizer.parameters[complexParamName], gaCurve_mean, label=str(optimizer.parameters['mutation_prob'][mut]))
                plt.fill_between(optimizer.parameters[complexParamName], gaCurve_mean - gaCurve_std,
                                 gaCurve_mean + gaCurve_std, alpha=0.25)

            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration w.r.t. Mutation for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            if problemName == 'One Max' or problemName == 'Continuous Peaks':
                plt.xlim(0, 16)
            elif problemName == 'Knapsack':
                plt.xlim(0, 20)
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, 2 ' + problemName + '.png')
            plt.close()

        Curve = []
        Time = []

        if optimizer.getOptimizerName() == 'MIMIC':
            for pop in range(len(optimizer.parameters['pop_size'])):
                for i in range(numerOfTrialsForGAandMIMIC):
                    _, state, curve = mlrose.mimic(problem,
                                                   pop_size=int(optimizer.parameters['pop_size'][pop]),
                                                   keep_pct=optimizer.keep_pct,
                                                   max_attempts=optimizer.max_attempts,
                                                   max_iters=int(max(optimizer.parameters['max_iters'])),
                                                   curve=True,
                                                   state_fitness_callback=time_callback,
                                                   callback_user_info=[])

                    curve = np.array(curve)

                    if len(curve) < int(max(optimizer.parameters['max_iters'])):
                        for i in range(int(int(max(optimizer.parameters['max_iters'])) - len(curve))):
                            curve = np.append(curve, curve[-1])

                    Curve.append(curve)
                    Time.append(times)

                mimicCurve_mean = np.array(Curve).mean(axis=0)
                mimicCurve_std = np.array(Curve).std(axis=0)
                complexParamName = 'max_iters'

                plt.style.use('seaborn-whitegrid')
                plt.plot(optimizer.parameters[complexParamName], mimicCurve_mean, label=str(optimizer.parameters['pop_size'][pop]))
                plt.fill_between(optimizer.parameters[complexParamName], mimicCurve_mean - mimicCurve_std,
                                 mimicCurve_mean + mimicCurve_std, alpha=0.25)

            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration w.r.t. Population Size for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            if problemName == 'One Max' or problemName == 'Continuous Peaks':
                plt.xlim(0, 11)
            elif problemName == 'Knapsack':
                plt.xlim(0, 16)
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, 1 ' + problemName + '.png')
            plt.close()

            Curve = []
            Time = []

            for pct in range(len(optimizer.parameters['keep_pct'])):
                for i in range(numerOfTrialsForGAandMIMIC):
                    _, state, curve = mlrose.mimic(problem,
                                                   pop_size=optimizer.pop_size,
                                                   keep_pct=optimizer.parameters['keep_pct'][pct],
                                                   max_attempts=optimizer.max_attempts,
                                                   max_iters=int(max(optimizer.parameters['max_iters'])),
                                                   curve=True,
                                                   state_fitness_callback=time_callback,
                                                   callback_user_info=[])

                    curve = np.array(curve)

                    if len(curve) < int(max(optimizer.parameters['max_iters'])):
                        for i in range(int(int(max(optimizer.parameters['max_iters'])) - len(curve))):
                            curve = np.append(curve, curve[-1])

                    Curve.append(curve)
                    Time.append(times)

                mimcCurve_mean = np.array(Curve).mean(axis=0)
                mimicCurve_std = np.array(Curve).std(axis=0)
                complexParamName = 'max_iters'

                plt.style.use('seaborn-whitegrid')
                plt.plot(optimizer.parameters[complexParamName], mimcCurve_mean, label=str(optimizer.parameters['keep_pct'][pct]))
                plt.fill_between(optimizer.parameters[complexParamName], mimcCurve_mean - mimicCurve_std,
                                 mimcCurve_mean + mimicCurve_std, alpha=0.25)

            plt.ylabel('Fitness', fontsize=12)
            plt.xlabel(complexParamName, fontsize=12)
            plt.title('Fitness vs Iteration w.r.t. Keep Pct for ' + optimizer.getOptimizerName(), fontsize=12, y=1.03)
            plt.legend()
            if problemName == 'One Max' or problemName == 'Continuous Peaks':
                plt.xlim(0, 11)
            elif problemName == 'Knapsack':
                plt.xlim(0, 16)
            plt.savefig('Figures/' + optimizer.getOptimizerName() + '-Fitness vs Iteration, 2 ' + problemName + '.png')
            plt.close()

    def getComparisonCurve(self, optimizers, problem, problemName):
        complexParamName = 'max_iters'
        numerOfTrials = 10
        numerOfTrialsForGAandMIMIC = 5
        Curve = []
        Time = []

        optimizer = optimizers[0]
        maxMaxAttempt = int(max(optimizer.parameters['max_iters']))
        for i in range(numerOfTrials):
            _, state, curve = mlrose.random_hill_climb(problem,
                                                       max_attempts=optimizer.max_attempts,
                                                       max_iters=int(max(optimizer.parameters['max_iters'])),
                                                       restarts=optimizer.restarts,
                                                       init_state=optimizer.init_state,
                                                       curve=True,
                                                       state_fitness_callback=time_callback,
                                                       random_state=optimizer.random_state,
                                                       callback_user_info=[])

            curve = np.array(curve)

            if len(curve) < maxMaxAttempt:
                for i in range(int(maxMaxAttempt - len(curve))):
                    curve = np.append(curve, curve[-1])

            Curve.append(curve)
            Time.append(times)

        rhcCurve_mean = np.array(Curve).mean(axis=0)
        rhcCurve_std = np.array(Curve).std(axis=0)
        rhcCurveTime = np.array(Time)
        Curve = []
        Time = []

        optimizer = optimizers[1]
        for i in range(numerOfTrials):
            _, state, curve = mlrose.simulated_annealing(problem,
                                                         schedule=optimizer.bestParameters['schedule'],
                                                         max_attempts=optimizer.max_attempts,
                                                         max_iters=int(max(optimizer.parameters['max_iters'])),
                                                         init_state=optimizer.init_state,
                                                         curve=True,
                                                         state_fitness_callback=time_callback,
                                                         random_state=optimizer.random_state,
                                                         callback_user_info=[])

            curve = np.array(curve)

            if len(curve) < maxMaxAttempt:
                for i in range(int(maxMaxAttempt - len(curve))):
                    curve = np.append(curve, curve[-1])

            Curve.append(curve)
            Time.append(times)

        saCurve_mean = np.array(Curve).mean(axis=0)
        saCurve_std = np.array(Curve).std(axis=0)
        saCurveTime = np.array(Time)
        Curve = []
        Time = []

        optimizer = optimizers[2]
        for i in range(numerOfTrialsForGAandMIMIC):
            _, state, curve = mlrose.genetic_alg(problem,
                                                 pop_size=int(optimizer.bestParameters['pop_size']),
                                                 mutation_prob=optimizer.bestParameters['mutation_prob'],
                                                 max_attempts=optimizer.max_attempts,
                                                 max_iters=int(max(optimizer.parameters['max_iters'])),
                                                 curve=True,
                                                 state_fitness_callback=time_callback,
                                                 random_state=optimizer.random_state,
                                                 callback_user_info=[])

            curve = np.array(curve)

            if len(curve) < maxMaxAttempt:
                for i in range(int(maxMaxAttempt - len(curve))):
                    curve = np.append(curve, curve[-1])

            Curve.append(curve)
            Time.append(times)

        gaCurve_mean = np.array(Curve).mean(axis=0)
        gaCurve_std = np.array(Curve).std(axis=0)
        gaCurveTime = np.array(Time)
        Curve = []
        Time = []

        optimizer = optimizers[3]
        for i in range(numerOfTrialsForGAandMIMIC):
            _, state, curve = mlrose.mimic(problem,
                                           pop_size=int(optimizer.bestParameters['pop_size']),
                                           keep_pct=optimizer.bestParameters['keep_pct'],
                                           max_attempts=optimizer.max_attempts,
                                           max_iters=int(max(optimizer.parameters['max_iters'])),
                                           curve=True,
                                           state_fitness_callback=time_callback,
                                           callback_user_info=[])

            curve = np.array(curve)

            if len(curve) < maxMaxAttempt:
                for i in range(int(maxMaxAttempt - len(curve))):
                    curve = np.append(curve, curve[-1])

            Curve.append(curve)
            Time.append(times)

        mimicCurve_mean = np.array(Curve).mean(axis=0)
        mimicCurve_std = np.array(Curve).std(axis=0)
        mimicCurveTime = np.array(Time)

        optimizer = optimizers[0]
        plt.style.use('seaborn-whitegrid')
        plt.plot(optimizer.parameters[complexParamName], rhcCurve_mean, label=str('Randomized Hill Climbing'))
        plt.fill_between(optimizer.parameters[complexParamName], rhcCurve_mean - rhcCurve_std,
                         rhcCurve_mean + rhcCurve_std, alpha=0.25)

        plt.plot(optimizer.parameters[complexParamName], saCurve_mean, label=str('Simulated Annealing'))
        plt.fill_between(optimizer.parameters[complexParamName], saCurve_mean - saCurve_std,
                         saCurve_mean + saCurve_std, alpha=0.25)

        plt.plot(optimizer.parameters[complexParamName], gaCurve_mean, label=str('Genetic Algorithm'))
        plt.fill_between(optimizer.parameters[complexParamName], gaCurve_mean - gaCurve_std,
                         gaCurve_mean + gaCurve_std, alpha=0.25)

        plt.plot(optimizer.parameters[complexParamName], mimicCurve_mean, label=str('MIMIC'))
        plt.fill_between(optimizer.parameters[complexParamName], mimicCurve_mean - mimicCurve_std,
                         mimicCurve_mean + mimicCurve_std, alpha=0.25)

        plt.ylabel('Fitness', fontsize=12)
        plt.xlabel(complexParamName, fontsize=12)
        plt.title('Optimizers Comparison for ' + problemName, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Optimizers Comparison for ' + problemName + '.png')
        plt.close()

        fig, ax = plt.subplots()
        plt.style.use('seaborn-whitegrid')
        x = ['RHC', 'SA', 'GA', 'MIMIC']
        time = [np.amax(np.amax(rhcCurveTime)), np.amax(np.amax(saCurveTime)),
                np.amax(np.amax(gaCurveTime)), np.amax(np.amax(mimicCurveTime))]
        x_pos = [i for i, _ in enumerate(x)]
        bars = plt.bar(x_pos, time)
        for idx, rect in enumerate(bars):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 4., (1.0 * height) + 0.1, '%f' % float(height))
        plt.ylabel('Time (s)', fontsize=12)
        plt.xlabel('Optimizers', fontsize=12)
        plt.xticks(x_pos, x)
        plt.title('Optimizers Convergence Time Comparison for ' + problemName, fontsize=12, y=1.03)
        plt.legend()
        plt.savefig('Figures/Optimizers Time Comparison for ' + problemName + '.png')
        plt.close()
