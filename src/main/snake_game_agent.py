import random

import numpy
import pygame
from numpy import ndarray

from consts import Consts, Colors, Action
from direction import Direction
from structs import Point

pygame.init()


class Game:

	def __init__(self, width=Consts.DEFAULT_WINDOW_WIDTH, height=Consts.DEFAULT_WINDOW_HEIGHT):
		self.grid_width = width // Consts.TILE_SIZE
		self.grid_height = height // Consts.TILE_SIZE

		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("arial", 25)
		self.display = pygame.display.set_mode(size=(width, height))
		pygame.display.set_caption("Snake Game")

		# Defining fields initialized in the reset() method:
		self.game_over = None
		self.score = None
		self.score_count = None
		self.snake = None
		self.front = None
		self.facing = None

		self.reset()

	def reset(self):
		# Initializing snake:
		self.facing = Direction.get_random_direction()
		self.front, self.snake = self.create_snake()

		self.score_count = 0
		self.score = None
		self.game_over = False
		self.place_score()

	def create_snake(self):
		front = Point(self.grid_width // 2, self.grid_height // 2)
		if self.facing == Direction.UP:
			snake = [front, Point(front.x, front.y + 1), Point(front.x, front.y + 2)]
		elif self.facing == Direction.DOWN:
			snake = [front, Point(front.x, front.y - 1), Point(front.x, front.y - 2)]
		elif self.facing == Direction.LEFT:
			snake = [front, Point(front.x + 1, front.y), Point(front.x + 2, front.y)]
		else:
			snake = [front, Point(front.x - 1, front.y), Point(front.x - 2, front.y)]
		return front, snake

	def place_score(self):
		while True:
			self.score = Point(random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
			if self.score not in self.snake:
				break

	def loop_iteration(self, action: Action) -> (int, bool, int):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit(0)

		reward = self.advance_snake(action)
		reward = self.collision_check(reward)

		self.render()
		self.clock.tick(Consts.FPS_AGENT)
		return reward, self.game_over, self.score_count

	def advance_snake(self, action: Action) -> int:
		# Expecting action to be a list representing [no turn, right turn, left turn].
		directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]  # Sorted clockwise.
		index = directions.index(self.facing)

		if action == Action.FORWARD:
			self.facing = directions[index]  # Making no turn.
		elif action == Action.LEFT:
			self.facing = directions[(index - 1) % 4]  # Making a left turn
		else:  # Action.RIGHT
			self.facing = directions[(index + 1) % 4]  # Making a right turn.

		x = self.front.x
		y = self.front.y

		if self.facing == Direction.RIGHT:
			x += 1
		elif self.facing == Direction.LEFT:
			x -= 1
		elif self.facing == Direction.DOWN:
			y += 1
		elif self.facing == Direction.UP:
			y -= 1

		self.front = Point(x, y)
		self.snake.insert(0, self.front)

		reward = 0
		if self.front == self.score:
			self.score_count += 1
			self.place_score()
			reward = 10
		else:
			self.snake.pop()

		return reward

	def collision_check(self, reward: int):
		self.game_over = self.collides(self.front)
		return -10 if self.game_over else reward

	def collides(self, point: Point) -> bool:
		return (
			# Checking collision at game edges:
				point.x > self.grid_width - 1 or point.x < 0 or point.y > self.grid_height - 1 or point.y < 0) or (
			# Checking if the snake collided with itself:
				point in self.snake[1:])

	def render(self):
		self.display.fill(Colors.BACKGROUND)

		for node in self.snake:
			pygame.draw.rect(self.display, Colors.SNAKE_OUTLINE,
							 pygame.Rect(node.x * Consts.TILE_SIZE, node.y * Consts.TILE_SIZE, Consts.TILE_SIZE,
										 Consts.TILE_SIZE))
			pygame.draw.rect(self.display, Colors.SNAKE_FILL,
							 pygame.Rect(node.x * Consts.TILE_SIZE + 4, node.y * Consts.TILE_SIZE + 4,
										 Consts.TILE_SIZE - 8, Consts.TILE_SIZE - 8))

		pygame.draw.rect(self.display, Colors.SCORE,
						 pygame.Rect(self.score.x * Consts.TILE_SIZE, self.score.y * Consts.TILE_SIZE, Consts.TILE_SIZE,
									 Consts.TILE_SIZE))

		text = self.font.render("Score: " + str(self.score_count), True, Colors.TEXT)
		self.display.blit(text, [0, 0])
		pygame.display.flip()

	def get_state(self) -> ndarray:
		front = self.front
		point_l = Point(front.x - 1, front.y)
		point_r = Point(front.x + 1, front.y)
		point_u = Point(front.x, front.y - 1)
		point_d = Point(front.x, front.y + 1)

		facing_l = self.facing == Direction.LEFT
		facing_r = self.facing == Direction.RIGHT
		facing_u = self.facing == Direction.UP
		facing_d = self.facing == Direction.DOWN

		state = [
			# Danger ahead (Relative to game.facing)
			(facing_r and self.collides(point_r)) or
			(facing_l and self.collides(point_l)) or
			(facing_u and self.collides(point_u)) or
			(facing_d and self.collides(point_d)),

			# Danger on the right side (Relative to game.facing)
			(facing_u and self.collides(point_r)) or
			(facing_d and self.collides(point_l)) or
			(facing_l and self.collides(point_u)) or
			(facing_r and self.collides(point_d)),

			# Danger on the left side (Relative to game.facing)
			(facing_d and self.collides(point_r)) or
			(facing_u and self.collides(point_l)) or
			(facing_r and self.collides(point_u)) or
			(facing_l and self.collides(point_d)),

			# Direction of movement
			facing_l,
			facing_r,
			facing_u,
			facing_d,

			# Score location.
			self.score.x < front.x,  # Score left
			self.score.x > front.x,  # Score right
			self.score.y < front.y,  # Score up
			self.score.y > front.y  # Score down
		]

		return numpy.array(state, dtype=int)
