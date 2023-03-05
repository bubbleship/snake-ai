import random

import pygame

from consts import Consts, Colors
from direction import Direction
from structs import Point


class Game:

	def __init__(self, width=Consts.DEFAULT_WINDOW_WIDTH, height=Consts.DEFAULT_WINDOW_HEIGHT):
		self.width = width
		self.height = height
		self.grid_width = width // Consts.TILE_SIZE
		self.grid_height = height // Consts.TILE_SIZE

		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont("arial", 25)
		self.display = pygame.display.set_mode(size=(width, height))
		pygame.display.set_caption("Snake Game")

		# Initializing snake:
		self.facing = Direction.get_random_direction()
		self.front, self.snake = self.create_snake()

		self.score_count = 0
		self.score = None
		self.input_processed = False
		self.keep_game_loop = True
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

	def loop(self):
		while self.keep_game_loop:
			self.input_processed = False
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.keep_game_loop = False
				self.process_input(event)

			self.advance_snake()
			self.collision_check()

			self.render()
			self.clock.tick(Consts.FPS)

		pygame.quit()
		print("Final Score: " + str(self.score_count))

	def process_input(self, event: pygame.event.Event):
		if self.input_processed:
			return
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT and self.facing != Direction.RIGHT:
				self.input_processed = True
				self.facing = Direction.LEFT
			elif event.key == pygame.K_RIGHT and self.facing != Direction.LEFT:
				self.input_processed = True
				self.facing = Direction.RIGHT
			elif event.key == pygame.K_UP and self.facing != Direction.DOWN:
				self.input_processed = True
				self.facing = Direction.UP
			elif event.key == pygame.K_DOWN and self.facing != Direction.UP:
				self.input_processed = True
				self.facing = Direction.DOWN

	def advance_snake(self):
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

		if self.front == self.score:
			self.score_count += 1
			self.place_score()
		else:
			self.snake.pop()

	def collision_check(self):
		# Checking collision at game edges:
		if self.front.x > self.grid_width or self.front.x < 0 or self.front.y > self.grid_height or self.front.y < 0:
			self.keep_game_loop = False
		# Checking if the snake collided with itself:
		if self.front in self.snake[1:]:
			self.keep_game_loop = False

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


def run():
	pygame.init()
	Game().loop()
