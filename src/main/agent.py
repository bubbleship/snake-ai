import json
import os
import random
from collections import deque

import numpy
import torch
from numpy import ndarray
from torch import optim, device
from torch.nn import MSELoss
from torch.optim import Adam

from consts import Consts, Action, AgentDataNames as ADN
from model import DQN
from structs import Transition


class Agent:
	gamma: float
	epsilon: float
	epsilon_min: float
	epsilon_decay: float
	device: device
	model: DQN
	optimizer: Adam
	criterion: MSELoss

	def __init__(self, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.005,
				 epsilon_decay: float = 0.9):
		"""
		:param gamma: Determines the importance of future rewards when
			calculating targets values
		:type gamma: float
		:param epsilon: The initial probability of this agent returning a
			random action via get_action()
		:type epsilon: float
		:param epsilon_min: The lowest probability of this agent to ever return
			a random action via get_action()
		:type epsilon_min: float
		:param epsilon_decay: The rate by which the probability of this agent
			choosing a random action is decreasing
		:type epsilon_decay: float
		"""
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = DQN(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_HIDDEN_LAYER_SIZE,
						 Consts.MODEL_OUTPUT_LAYER_SIZE).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=Consts.LEARNING_RATE)
		self.criterion = MSELoss()

	def get_action(self, state: ndarray) -> Action:
		"""
		Selects an action in an epsilon-greedy approach meaning, a random
		action is selected at probability epsilon and an action based on the
		given state at probability 1-epsilon

		:param state: An array representing the current state of the
			environment
		:type state: ndarray
		:return: a random action at a probability of epsilon or an action based
			on the given state at a probability of 1-epsilon
		:rtype: Action
		"""
		action: Action

		if torch.rand(1) < self.epsilon:  # Exploring environment.
			action = Action(random.randint(0, len(Action) - 1))
		else:  # Making prediction.
			with torch.inference_mode():
				state = torch.tensor(state).unsqueeze(0).float().to(self.device)
				q_values = self.model(state)
				action = Action(q_values.argmax(1).item())

		return action

	def train(self, memory: deque):
		if len(memory) > Consts.BATCH_SIZE:
			sample = random.sample(memory, Consts.BATCH_SIZE)
		else:
			sample = memory

		batch = Transition(*zip(*sample))
		previous_state = numpy.stack(
			batch.previous_state)  # Creating a tensor from a list of numpy.ndarray objects is slow.
		next_state = numpy.stack(
			batch.next_state)  # Creating a tensor from a list of numpy.ndarray objects is slow.
		previous_state_tensor = torch.tensor(previous_state, dtype=torch.float, device=self.device)
		action_tensor = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(-1)
		reward_tensor = torch.tensor(batch.reward, dtype=torch.float, device=self.device).unsqueeze(-1)
		next_state_tensor = torch.tensor(next_state, dtype=torch.float, device=self.device)
		is_game_over_tensor = torch.tensor(batch.is_game_over, dtype=torch.int, device=self.device).unsqueeze(-1)

		max_q_values = self.model(next_state_tensor).max(dim=1, keepdim=True)[0]
		targets = (reward_tensor + (self.gamma * max_q_values * (1 - is_game_over_tensor)))

		q_values = self.model(previous_state_tensor)
		action_q_values = q_values.gather(dim=1, index=action_tensor)

		self.optimizer.zero_grad()
		loss = self.criterion(action_q_values, targets)
		loss.backward()
		self.optimizer.step()

	def decay_epsilon(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def save(self):
		self.model.save()

		data = {
			ADN.EPSILON: self.epsilon
		}

		with open(os.path.join(Consts.MODEL_DIR_PATH, Consts.AGENT_DATA_FILE_NAME), "w") as file:
			json.dump(data, file, indent=2)

	def load(self):
		self.model.load()

		path = os.path.join(Consts.MODEL_DIR_PATH, Consts.AGENT_DATA_FILE_NAME)
		if not os.path.exists(path):
			return

		with open(path, "r") as file:
			data = json.load(file)

		self.epsilon = data[ADN.EPSILON]
