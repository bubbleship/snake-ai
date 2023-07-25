import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential

from commons.consts import Consts


class DQN(nn.Module):
	layers: Sequential

	def __init__(self, state_dim: int, hidden_dim: int, action_dim):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(state_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, action_dim)
		)

	def forward(self, x: Tensor) -> Tensor:
		return self.layers(x)

	def save(self):
		if not os.path.exists(Consts.MODEL_DIR_PATH):
			os.makedirs(Consts.MODEL_DIR_PATH)

		path = os.path.join(Consts.MODEL_DIR_PATH, Consts.MODEL_FILE_NAME)
		torch.save(self.state_dict(), path)

	def load(self):
		path = os.path.join(Consts.MODEL_DIR_PATH, Consts.MODEL_FILE_NAME)
		if not os.path.exists(path):
			return

		self.load_state_dict(torch.load(path))
