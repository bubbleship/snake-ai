import random
from enum import Enum


class Direction(Enum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3

	@staticmethod
	def get_random_direction():
		return Direction(random.randint(a=0, b=3))
