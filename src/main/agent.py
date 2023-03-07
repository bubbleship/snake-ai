from collections import deque

from numpy import ndarray

from main.consts import Consts


class Agent:

	def __init__(self):
		self.games_count = 0
		self.epsilon = 0
		self.gamma = 0
		self.memory = deque(maxlen=Consts.MAX_MEMORY)

	def remember(self, previous_state: ndarray, action: list[3], reward: int, next_state: ndarray,
				 is_game_over: bool) -> None:
		self.memory.append((previous_state, action, reward, next_state, is_game_over))
