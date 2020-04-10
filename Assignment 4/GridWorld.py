"""
Course:         CS 7641 Assignment 4, Spring 2020
Date:           March 31st, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
Comment:        Almost all functions defined in this file for creating the grid world problem to transition and
reward functions calculations, except the getActions, are stolen from the following GitHub repo:
https://github.com/jrrockett/ML-MDP/blob/16830855b109717d4e9d56e6f96fbcd1e35b5dbf/frozen_lake/frozen_lake_mdp.py
"""

import gridworld as gv
import numpy as np


class FrozenLake:
	def __init__(self, data, size, holes_coord, start, goal, random_state):
		self.random_state = random_state
		self.size = size
		self.holes_coord = holes_coord
		self.start = start
		self.goal = goal
		self.lake_init(random_state)
		self.data = data

	def lake_init(self, random_state=None):
		self.lake = [[0 for i in range(self.size)] for j in range(self.size)]
		if random_state:
			np.random.seed(random_state)
		for i in range(self.size):
			for j in range(self.size):
				if (i, j) in self.holes_coord:
					self.lake[i][j] = -100
				else:
					self.lake[i][j] = -1
		self.lake[self.goal[0]][self.goal[1]] = 10

	def tile2classes(self, x, y):
		holesCoords = []
		for row in range(0, self.data.shape[0]):
			for col in range(0, self.data.shape[1]):
				if self.data[row, col] == 1:  # Obstacle
					holesCoords.append((row, col))
				if self.data[row, col] == 2:  # El roboto
					start = (row, col)
				if self.data[row, col] == 3:  # Goal
					goal = (row, col)

		if (x, y) in holesCoords:
			return "water"
		elif (x, y) == goal:
			return "goal"

		return "normal"


def create_transition_matrix(size, move_prob):
	n_states = size*size
	actions = [(0, -1), (-1, 0), (0, 1), (1, 0)] #L, U, R, D
	transitions = np.zeros((4, n_states, n_states))
	for k in range(len(actions)):
		for i in range(size):
			for j in range(size):
				state0_ind = i*size + j
				for d in range(-1, 2):
					action = actions[(k + d)%4]
					state1_ind = (i + action[0])*size + j + action[1]
					if (k == 0 or k == 2) and d == 0 and (state1_ind >= (i+1)*size or state1_ind < i*size) and state1_ind >= 0 and state1_ind < n_states: #test for left right
						transitions[k][state0_ind][state0_ind] += move_prob
					elif (k == 1 or k==3) and d != 0 and (state1_ind >= (i+1)*size or state1_ind < i*size) and state1_ind >= 0 and state1_ind < n_states: #test up down
						transitions[k][state0_ind][state0_ind] += (1-move_prob)/2
					elif state1_ind >= 0 and state1_ind < n_states:
						if d == 0:
							transitions[k][state0_ind][state1_ind] += move_prob
						else:
							transitions[k][state0_ind][state1_ind] += (1-move_prob)/2
					else:
						if d == 0:
							transitions[k][state0_ind][state0_ind] += move_prob
						else:
							transitions[k][state0_ind][state0_ind] += (1-move_prob)/2

	return transitions


def create_reward_matrix(grid):
	reward = np.zeros((len(grid)**2))
	for i in range(len(grid)):
		for j in range(len(grid[0])):
			if grid[i][j] == -100:
				reward[i*len(grid)+j] = -100
			elif grid[i][j] == -1:
				reward[i*len(grid)+j] = -1
			elif grid[i][j] == 10:
				reward[i*len(grid)+j] = 100

	return reward


def print_as_grid(my_vec, lake, dim, policy=True):
	if policy:
		my_vec = list(my_vec)
		for i in range(len(my_vec)):
			if my_vec[i] == 0:
				my_vec[i] = '<'
			if my_vec[i] == 1:
				my_vec[i] = '^'
			if my_vec[i] == 2:
				my_vec[i] = '>'
			if my_vec[i] == 3:
				my_vec[i] = 'v'
			if lake[int(i/dim)][i%dim] == -100:
				my_vec[i] = 'O'
			if lake[int(i/dim)][i%dim] == 100:
				my_vec[i] = 'X'

	for i in range(dim):
		ind0 = i*dim
		ind1 = (i+1)*dim
		print(my_vec[ind0:ind1])
	print()


def print_policy_vs_value(p_policy, v_policy):
	print_as_grid(p_policy)
	print()
	print_as_grid(v_policy)


def get_environement(data, size, holesCoords, start, goal):
	lakeObject = FrozenLake(data, size, holesCoords, start, goal, random_state=15)
	lake = lakeObject.lake
	transitions = create_transition_matrix(len(lake), 0.8)
	reward = create_reward_matrix(lake)
	discount = .9

	return transitions, reward, discount, lakeObject

def getActions(policyList, start, goal, dim):
	notReached = True
	policy = []
	actions = []
	attemps = 1

	for i in range(dim):
		ind0 = i*dim
		ind1 = (i+1)*dim
		policy.append(policyList[ind0:ind1])

	if policy[start[0]][start[1]] == 0:
		next = (start[0], start[1] - 1)
		actions.append(gv.W)
	elif policy[start[0]][start[1]] == 1:
		next = (start[0] - 1, start[1])
		actions.append(gv.N)
	elif policy[start[0]][start[1]] == 2:
		next = (start[0], start[1] + 1)
		actions.append(gv.E)
	elif policy[start[0]][start[1]] == 3:
		next = (start[0] + 1, start[1])
		actions.append(gv.S)

	while notReached and attemps < dim**2:
		if policy[next[0]][next[1]] == 0:
			next = (next[0], next[1] - 1)
			actions.append(gv.W)
		elif policy[next[0]][next[1]] == 1:
			next = (next[0] - 1, next[1])
			actions.append(gv.N)
		elif policy[next[0]][next[1]] == 2:
			next = (next[0], next[1] + 1)
			actions.append(gv.E)
		elif policy[next[0]][next[1]] == 3:
			next = (next[0] + 1, next[1])
			actions.append(gv.S)

		if next == goal:
			notReached = False
		attemps += 1

	return actions
