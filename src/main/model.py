import os

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class LinearQNet(nn.Module):

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
