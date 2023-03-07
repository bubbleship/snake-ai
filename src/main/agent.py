from collections import deque

from consts import Consts


class Agent:

	def __init__(self):
		self.games_count = 0
		self.epsilon = 0
		self.gamma = 0
		self.memory = deque(maxlen=Consts.MAX_MEMORY)
