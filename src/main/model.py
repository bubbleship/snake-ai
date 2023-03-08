import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
from numpy import ndarray
from torch import Tensor, optim
from torch.nn import Linear, MSELoss
from torch.optim import Adam


class LinearQNet(nn.Module):
	input_to_hidden: Linear
	hidden_to_output: Linear

	def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size):
		super().__init__()
		self.input_to_hidden = nn.Linear(input_layer_size, hidden_layer_size)
		self.hidden_to_output = nn.Linear(hidden_layer_size, output_layer_size)

	def forward(self, x: Tensor) -> Tensor:
		x = functional.relu(self.input_to_hidden(x))
		x = self.hidden_to_output(x)
		return x

	def save(self, file_name: str = "model.pth"):
		model_directory_path = "./model"
		if not os.path.exists(model_directory_path):
			os.makedirs(model_directory_path)

		path = os.path.join(model_directory_path, file_name)
		torch.save(self.state_dict(), path)


class QTrainer:
	model: LinearQNet
	gamma: float
	optimizer: Adam
	criterion: MSELoss

	def __init__(self, model: LinearQNet, lr: float, gamma: float):
		self.model = model
		self.gamma = gamma
		self.optimizer = optim.Adam(model.parameters(), lr)
		self.criterion = nn.MSELoss()

	def train_step(self, previous_state: ndarray, action: list[3], reward: int, next_state: ndarray,
				   is_game_over: bool) -> None:
		previous_state = torch.tensor(previous_state, dtype=torch.float)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor(action, dtype=torch.long)
		reward = torch.tensor(reward, dtype=torch.float)

		if len(previous_state.shape) == 1:
			previous_state = torch.unsqueeze(previous_state, 0)
			next_state = torch.unsqueeze(next_state, 0)
			action = torch.unsqueeze(action, 0)
			reward = torch.unsqueeze(reward, 0)
			is_game_over = (is_game_over,)

		prediction = self.model(previous_state)

		target = prediction.clone()
		for i in range(len(is_game_over)):
			new_q = reward[i]
			if not is_game_over[i]:
				new_q = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

			target[i][torch.argmax(action[i]).item()] = new_q

		self.optimizer.zero_grad()
		loss = self.criterion(target, prediction)
		loss.backward()

		self.optimizer.step()
