from typing import NamedTuple

from numpy import ndarray


class Point(NamedTuple):
	x: int
	y: int


class Transition(NamedTuple):
	previous_state: ndarray
	action: int
	reward: int
	next_state: ndarray
	is_game_over: bool
