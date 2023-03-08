import torch.nn as nn

class LinearQNet(nn.Module):

	def __init__(self, input_layer_size: int, hidden_layer_size: int, output_layer_size):
		super().__init__()
		self.input_to_hidden = nn.Linear(input_layer_size, hidden_layer_size)
		self.hidden_to_output = nn.Linear(hidden_layer_size, output_layer_size)
