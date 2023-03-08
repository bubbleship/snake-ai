from collections import deque

from numpy import ndarray

from main.consts import Consts
from main.model import LinearQNet, QTrainer


class Agent:

	def __init__(self):
		self.games_count = 0
		self.epsilon = 0
		self.gamma = 0.9  # Discount rate
		self.memory = deque(maxlen=Consts.MAX_MEMORY)
		self.model = LinearQNet(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_HIDDEN_LAYER_SIZE,
								Consts.MODEL_OUTPUT_LAYER_SIZE)
		self.trainer = QTrainer(self.model, Consts.LEARNING_RATE, self.gamma)

	def remember(self, previous_state: ndarray, action: list[3], reward: int, next_state: ndarray,
				 is_game_over: bool) -> None:
		self.memory.append((previous_state, action, reward, next_state, is_game_over))
