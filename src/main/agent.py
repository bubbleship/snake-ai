import random

import numpy
import torch
from numpy import ndarray
from torch import optim, device
from torch.nn import MSELoss
from torch.optim import Adam

from consts import Consts, Action
from model import DQN
from structs import Transition


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
		self.optimizer = optim.Adam(self.model.parameters(), lr=Consts.LEARNING_RATE)
		self.criterion = MSELoss()

	def get_action(self, state: ndarray) -> Action:
		action: Action

		if torch.rand(1) < self.epsilon: # Exploring environment.
			action = Action(random.randint(0, len(Action) - 1))
		else: # Making prediction.
			with torch.no_grad():
				state = torch.tensor(state).unsqueeze(0).float().to(self.device)
				q_values = self.model(state)
				action = Action(q_values.argmax(1).item())

		return action

	def train(self, memory):
		if len(memory) > Consts.BATCH_SIZE:
			sample = random.sample(memory, Consts.BATCH_SIZE)
		else:
			sample = memory

		#previous_state_batch, action_batch, reward_batch, next_state_batch, is_game_over_batch = zip(*sample)
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
		loss = self.criterion(action_q_values, targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
