"""
Course:         CS 7641 Assignment 4, Spring 2020
Date:           March 31st, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

from hiive.mdptoolbox import mdp
from GridWorld import *
from Forest import *
import GridWorldVisualizer as gv
import matplotlib.pyplot as plt
import numpy as np
import QLearner


def findBestPolicyForGridWorlds(worlds, grid, starts, goals):
	qlearningIter = [1000, 10000]
	worldCntr = 1

	for data in worlds:
		size = len(data)
		holesCoords = []
		for row in range(0, data.shape[0]):
			for col in range(0, data.shape[1]):
				if data[row, col] == 1:  # Obstacle
					holesCoords.append((row, col))
				if data[row, col] == 2:  # El roboto
					start = (row, col)
				if data[row, col] == 3:  # Goal
					goal = (row, col)
		transitions, reward, discount, lake = get_environement(data, size, holesCoords, start, goal)

		#Policy iteration
		policy_iteration = mdp.PolicyIteration(transitions, reward, discount, policy0=None, max_iter=1000, eval_type=0)
		policy_iteration.run()
		print_as_grid(policy_iteration.policy, lake.lake, size)
		print(policy_iteration.time)
		print(policy_iteration.iter)

		actions = getActions(policy_iteration.policy, start, goal, size)
		svg = gv.gridworld(n=size, tile2classes=lake.tile2classes, actions=actions, extra_css='goal', start=start, policyList=policy_iteration.policy)
		svg.saveas("Figures/Grid/PI-Final-Path for World "+str(worldCntr)+".svg", pretty=True)

		#Value iteration
		value_iteration = mdp.ValueIteration(transitions, reward, discount, epsilon=0.001, max_iter=1000, initial_value=0)
		value_iteration.run()
		print_as_grid(value_iteration.policy, lake.lake, size)
		print(value_iteration.time)
		print(value_iteration.iter)

		actions = getActions(value_iteration.policy, start, goal, size)
		svg = gv.gridworld(n=size, tile2classes=lake.tile2classes, actions=actions, extra_css='goal', start=start, policyList=value_iteration.policy)
		svg.saveas("Figures/Grid/VI-Final-Path for World "+str(worldCntr)+".svg", pretty=True)

		#Q-Learning
		q_learning = QLearner.QLearningEx(transitions, reward, grid=grid[worldCntr-1], start=starts[worldCntr-1], goals=goals[worldCntr-1], n_iter=qlearningIter[worldCntr-1],
										  n_restarts=1000, alpha = 0.2, gamma = 0.9, rar = 0.1, radr = 0.99)
		q_learning.run()
		print_as_grid(q_learning.policy, lake.lake, size)
		#print(q_learning.time)

		actions = getActions(q_learning.policy, start, goal, size)
		svg = gv.gridworld(n=size, tile2classes=lake.tile2classes, actions=actions, extra_css='goal', start=start, policyList=q_learning.policy)
		svg.saveas("Figures/Grid/QL-Final-Path for World "+str(worldCntr)+".svg", pretty=True)

		worldCntr += 1

def defineGridWorlds():
	worlds = []
	grid = []
	goals = []
	starts = []
	filename = 'testworlds/world01.csv'
	inf = open(filename)
	data = np.array([list(map(np.float, s.strip().split(','))) for s in inf.readlines()])
	worlds.append(data)
	grid.append(np.zeros(shape=(data.shape[0], data.shape[1])))
	goals.append([4])
	starts.append(21)
	filename = 'testworlds/world02.csv'
	inf = open(filename)
	data = np.array([list(map(np.float, s.strip().split(','))) for s in inf.readlines()])
	worlds.append(data)
	grid.append(np.zeros(shape=(data.shape[0], data.shape[1])))
	goals.append([24])
	starts.append(607)

	return worlds, grid, starts, goals


def getPlotsForGridWorldViPi(worlds, grid, starts, goals):
	iters = []
	iter = range(1, 21, 1)
	iters.append(iter)
	iter = range(1, 41, 1)
	iters.append(iter)
	qlearningIter = [100000, 100000000]
	worldCntr = 1

	for data in worlds:
		pi_rewards = []
		pi_error = []
		pi_time = []
		pi_iter = []
		vi_rewards = []
		vi_error = []
		vi_time = []
		vi_iter = []
		size = len(data)
		holesCoords = []
		for row in range(0, data.shape[0]):
			for col in range(0, data.shape[1]):
				if data[row, col] == 1:  # Obstacle
					holesCoords.append((row, col))
				if data[row, col] == 2:  # El roboto
					start = (row, col)
				if data[row, col] == 3:  # Goal
					goal = (row, col)
		transitions, reward, discount, lake = get_environement(data, size, holesCoords, start, goal)

		for iter in iters[worldCntr-1]:
			# Policy iteration
			policy_iteration = mdp.PolicyIteration(transitions, reward, discount, policy0=None, max_iter=iter,
												   eval_type=0)
			policy_iteration.run()
			print_as_grid(policy_iteration.policy, lake.lake, size)
			pi_rewards.append(policy_iteration.run_stats[len(policy_iteration.run_stats)-1]['Reward'])
			pi_error.append(policy_iteration.run_stats[len(policy_iteration.run_stats)-1]['Error'])
			pi_time.append(policy_iteration.run_stats[len(policy_iteration.run_stats)-1]['Time'])
			pi_iter.append(policy_iteration.run_stats[len(policy_iteration.run_stats)-1]['Iteration'])

			# Value iteration
			value_iteration = mdp.ValueIteration(transitions, reward, discount, epsilon=0.001, max_iter=iter,
												 initial_value=0)
			value_iteration.run()
			print_as_grid(value_iteration.policy, lake.lake, size)
			vi_rewards.append(value_iteration.run_stats[len(value_iteration.run_stats)-1]['Reward'])
			vi_error.append(value_iteration.run_stats[len(value_iteration.run_stats)-1]['Error'])
			vi_time.append(value_iteration.run_stats[len(value_iteration.run_stats)-1]['Time'])
			vi_iter.append(value_iteration.run_stats[len(value_iteration.run_stats)-1]['Iteration'])

		plt.style.use('seaborn-whitegrid')
		plt.plot(iters[worldCntr-1], pi_error, label='PI')
		plt.plot(iters[worldCntr-1], vi_error, label='VI')
		plt.ylabel('Convergence', fontsize=12)
		plt.xlabel('Iter.', fontsize=12)
		plt.title('Convergence vs Iteration for Grid World no.' + str(worldCntr), fontsize=12, y=1.03)
		plt.legend()
		plt.savefig('Figures/Grid/Convergence vs Iteration for Grid World no.' + str(worldCntr) + '.png')
		plt.close()

		plt.style.use('seaborn-whitegrid')
		plt.plot(iters[worldCntr-1], pi_rewards, label='PI')
		plt.plot(iters[worldCntr-1], vi_rewards, label='VI')
		plt.ylabel('Reward', fontsize=12)
		plt.xlabel('Iter.', fontsize=12)
		plt.title('Reward vs Iteration for Grid World no.' + str(worldCntr), fontsize=12, y=1.03)
		plt.legend()
		plt.savefig('Figures/Grid/Reward vs Iteration for Grid World no.' + str(worldCntr) + '.png')
		plt.close()

		plt.style.use('seaborn-whitegrid')
		plt.plot(iters[worldCntr-1], pi_time, label='PI')
		plt.plot(iters[worldCntr-1], vi_time, label='VI')
		plt.ylabel('Time', fontsize=12)
		plt.xlabel('Iter.', fontsize=12)
		plt.title('Time vs Iteration for Grid World no.' + str(worldCntr), fontsize=12, y=1.03)
		plt.legend()
		plt.savefig('Figures/Grid/Time vs Iteration for Grid World no.' + str(worldCntr) + '.png')
		plt.close()

		worldCntr += 1


def getPlotsForGridWorldQl(worlds, grid, starts, goals):
	iters = range(1, 21, 1)
	lRates = [x for x in [0.2, 0.7]]
	epsilons = [x for x in [0.1, 0.9]]
	qlearningIter = [1000, 10000]
	worldCntr = 1

	for data in worlds:
		ql_rewards = []
		ql_error = []
		ql_time = []
		ql_iter = []
		size = len(data)
		holesCoords = []
		for row in range(0, data.shape[0]):
			for col in range(0, data.shape[1]):
				if data[row, col] == 1:  # Obstacle
					holesCoords.append((row, col))
				if data[row, col] == 2:  # El roboto
					start = (row, col)
				if data[row, col] == 3:  # Goal
					goal = (row, col)
		transitions, reward, discount, lake = get_environement(data, size, holesCoords, start, goal)

		for lRate in lRates:
			for epsilon in epsilons:
				# Q-Learning
				q_learning = QLearner.QLearningEx(transitions, reward, grid=grid[worldCntr - 1],
													start=starts[worldCntr - 1], goals=goals[worldCntr - 1],
													n_iter=qlearningIter[worldCntr - 1],
													n_restarts=1000, alpha=lRate, gamma=0.9, rar=epsilon, radr=0.99)
				q_learning.run()

				q_learning.run()
				print_as_grid(q_learning.policy, lake.lake, size)
				ql_rewards.append(q_learning.episode_reward)
				ql_time.append(q_learning.episode_times)
				ql_error.append(q_learning.episode_error)

		elCntr = 0
		run_stat_frequency = max(1, qlearningIter[worldCntr-1] // 10000)

		print("First Combination reward mean: ", np.mean(ql_rewards[0]))
		print("Second Combination reward mean: ", np.mean(ql_rewards[1]))
		print("Third Combination reward mean: ", np.mean(ql_rewards[2]))
		print("Four Combination reward mean: ", np.mean(ql_rewards[3]))
		print("First Combination error mean: ", np.mean(ql_error[0]))
		print("Second Combination error mean: ", np.mean(ql_error[1]))
		print("Third Combination error mean: ", np.mean(ql_error[2]))
		print("Four Combination error mean: ", np.mean(ql_error[3]))

		plt.figure(figsize=(15, 8))
		plt.style.use('seaborn-whitegrid')
		for lRate in lRates:
			for epsilon in epsilons:
				if lRate == 0.2:
					plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10], label='a: '+str(lRate)+', e: '+str(epsilon))
					elCntr += 1
				else:
					plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10],
							 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
					elCntr += 1
		plt.ylabel('Convergence', fontsize=12)
		plt.xlabel('Iter. (x'+str(qlearningIter[worldCntr-1])+')', fontsize=12)
		plt.title('Convergence vs Iteration for Grid World no.' + str(worldCntr), fontsize=12, y=1.03)
		plt.legend()
		plt.savefig('Figures/Grid/Convergence vs Iteration for Grid World no.' + str(worldCntr) + ', QL.png')
		plt.close()

		elCntr = 0

		plt.figure(figsize=(15, 8))
		plt.style.use('seaborn-whitegrid')
		for lRate in lRates:
			for epsilon in epsilons:
				if lRate == 0.2:
					plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
							 label='a: ' + str(lRate) + ', e: ' + str(epsilon))
				else:
					plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
							 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
				elCntr += 1
		plt.ylabel('Reward', fontsize=12)
		plt.xlabel('Iter. (x'+str(qlearningIter[worldCntr-1])+')', fontsize=12)
		plt.title('Reward vs Iteration for Grid World no.' + str(worldCntr), fontsize=12, y=1.03)
		plt.legend()
		plt.savefig('Figures/Grid/Reward vs Iteration for Grid World no.' + str(worldCntr) + ', QL.png')
		plt.close()

		worldCntr += 1


def findBestPolicyForForest():
	cntr = 0
	pi_rewards = []
	pi_error = []
	pi_time = []
	pi_iter = []
	vi_rewards = []
	vi_error = []
	vi_time = []
	vi_iter = []
	for size in [15]:
		forest = ForestMng(states=size, reward_wait=4, reward_cut=2)

		# Policy iteration
		policy_iteration = mdp.PolicyIteration(forest.P, forest.R, gamma=0.9, policy0=None, max_iter=1000, eval_type=0)
		policy_iteration.run()
		print(policy_iteration.time)
		print(policy_iteration.iter)
		print(policy_iteration.policy)
		pi_rewards.append([sub['Reward'] for sub in policy_iteration.run_stats])
		pi_error.append([ sub['Error'] for sub in policy_iteration.run_stats ])
		pi_time.append([ sub['Time'] for sub in policy_iteration.run_stats ])
		pi_iter.append([ sub['Iteration'] for sub in policy_iteration.run_stats ])

		# Value iteration
		value_iteration = mdp.ValueIteration(forest.P, forest.R, gamma=0.9, max_iter=1000)
		value_iteration.run()
		print(value_iteration.time)
		print(value_iteration.iter)
		print(value_iteration.policy)
		vi_rewards.append([sub['Reward'] for sub in value_iteration.run_stats])
		vi_error.append([sub['Error'] for sub in value_iteration.run_stats])
		vi_time.append([sub['Time'] for sub in value_iteration.run_stats])
		vi_iter.append([sub['Iteration'] for sub in value_iteration.run_stats])

		if max(pi_iter[cntr]) < max(vi_iter[cntr]):
			for i in range(max(vi_iter[cntr]) - max(pi_iter[cntr])):
				pi_error[cntr].append(pi_error[cntr][len(pi_error[cntr])-1])
				pi_rewards[cntr].append(pi_rewards[cntr][len(pi_rewards[cntr]) - 1])
				pi_time[cntr].append(pi_time[cntr][len(pi_time[cntr]) - 1])

		cntr += 1

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_error[0], label='PI')
	plt.plot(vi_iter[0], vi_error[0], label='VI')
	plt.ylabel('Convergence', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Convergence vs Iteration for Forest Mng', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Convergence vs Iteration for Forest Mng.png')
	plt.close()

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_rewards[0], label='PI')
	plt.plot(vi_iter[0], vi_rewards[0], label='VI')
	plt.ylabel('Reward', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Rewards vs Iteration for Forest Mng', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Rewards vs Iteration for Forest Mng.png')
	plt.close()

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_time[0], label='PI')
	plt.plot(vi_iter[0], vi_time[0], label='VI')
	plt.ylabel('Time', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Time vs Iteration for Forest Mng', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Time vs Iteration for Forest Mng.png')
	plt.close()


def getPlotsForForestQl():
	iters = range(1, 21, 1)
	lRates = [x for x in [0.2, 0.7]]
	epsilons = [x for x in [0.1, 0.9]]
	ql_rewards = []
	ql_error = []
	ql_time = []
	ql_iter = []

	forest = ForestMng(states=15, reward_wait=4, reward_cut=2)

	for lRate in lRates:
		for epsilon in epsilons:
			# Q-Learning
			q_learning = QLearner.QLearningEx(forest.P, forest.R, grid=np.zeros(shape=(15, 1)), start=0, goals=[14],
											  n_iter=1000, n_restarts=1000, alpha=lRate, gamma=0.9, rar=epsilon,
											  radr=0.999999)
			q_learning.run()
			ql_rewards.append(q_learning.episode_reward)
			ql_time.append(q_learning.episode_times)
			ql_error.append(q_learning.episode_error)
			print(q_learning.policy)

	elCntr = 0

	print("First Combination reward mean: ", np.mean(ql_rewards[0]))
	print("Second Combination reward mean: ", np.mean(ql_rewards[1]))
	print("Third Combination reward mean: ", np.mean(ql_rewards[2]))
	print("Four Combination reward mean: ", np.mean(ql_rewards[3]))
	print("First Combination error mean: ", np.mean(ql_error[0]))
	print("Second Combination error mean: ", np.mean(ql_error[1]))
	print("Third Combination error mean: ", np.mean(ql_error[2]))
	print("Four Combination error mean: ", np.mean(ql_error[3]))

	plt.figure(figsize=(15, 8))
	plt.style.use('seaborn-whitegrid')
	for lRate in lRates:
		for epsilon in epsilons:
			if lRate == 0.2:
				plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon))
				elCntr += 1
			else:
				plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
				elCntr += 1
	plt.ylabel('Convergence', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Convergence vs Iteration for Forest Mng', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Convergence vs Iteration for Forest Mng, QL.png')
	plt.close()

	elCntr = 0

	plt.figure(figsize=(15, 8))
	plt.style.use('seaborn-whitegrid')
	for lRate in lRates:
		for epsilon in epsilons:
			if lRate == 0.2:
				plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon))
			else:
				plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
			elCntr += 1
	plt.ylabel('Reward', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Reward vs Iteration for Forest Mng', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Reward vs Iteration for Forest Mng, QL.png')
	plt.close()


def main():
	worlds, grid, starts, goals = defineGridWorlds()
	findBestPolicyForGridWorlds(worlds, grid, starts, goals)
	getPlotsForGridWorldViPi(worlds, grid, starts, goals)
	getPlotsForGridWorldQl(worlds, grid, starts, goals)
	findBestPolicyForForest()
	getPlotsForForestQl()


if __name__ == '__main__':
	main()
