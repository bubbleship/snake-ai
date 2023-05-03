import random

import numpy
import torch
from numpy import ndarray
from torch import optim, nn, device
from torch.nn import MSELoss
from torch.optim import Adam

from consts import Consts
from model import DQN


class Agent:
	state_dim: int
	epsilon_decay: float
	epsilon_min: float
	epsilon: float
	gamma: float
	action_dim: int
	device: device
	model: DQN
	optimizer: Adam
	criterion: MSELoss

	def __init__(self, state_dim: int, action_dim: int, gamma: float = 0.99, epsilon: float = 1.0,
				 epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = DQN(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_HIDDEN_LAYER_SIZE,
						 Consts.MODEL_OUTPUT_LAYER_SIZE).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters())
		self.criterion = nn.MSELoss()

	def get_action(self, state: ndarray) -> list[3]:
		action: list[int] = [0, 0, 0]

		if torch.rand(1) < self.epsilon: # Exploring environment.
			action_type = random.randint(0, len(action) - 1)
			action[action_type] = 1
		else: # Making prediction.
			with torch.no_grad():
				state = torch.tensor(state).unsqueeze(0).float().to(self.device)
				q_values = self.model(state)
				action_type = q_values.argmax(1).item()
				action[action_type] = 1

		return action

	def train(self, memory):
		if len(memory) > Consts.BATCH_SIZE:
			sample = random.sample(memory, Consts.BATCH_SIZE)
		else:
			sample = memory

		previous_state_batch, action_batch, reward_batch, next_state_batch, is_game_over_batch = zip(*sample)
		previous_state_batch = numpy.stack(
			previous_state_batch)  # Creating a tensor from a list of numpy.ndarray objects is slow.
		next_state_batch = numpy.stack(
			next_state_batch)  # Creating a tensor from a list of numpy.ndarray objects is slow.
		previous_state_tensor = torch.tensor(previous_state_batch, dtype=torch.float, device=self.device)
		action_tensor = torch.tensor(action_batch, dtype=torch.long, device=self.device)
		reward_tensor = torch.tensor(reward_batch, dtype=torch.float, device=self.device)
		next_state_tensor = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
		is_game_over_tensor = torch.tensor(is_game_over_batch, dtype=torch.int, device=self.device)

		q_values = self.model(previous_state_tensor)
		q_values = torch.sum(q_values * action_tensor, dim=1)
		next_q_values = self.model(next_state_tensor).max(1)[0]
		expected_q_values = (reward_tensor + (self.gamma * next_q_values * (1 - is_game_over_tensor)))

		loss = self.criterion(q_values, expected_q_values)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
