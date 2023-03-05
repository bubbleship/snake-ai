import pygame

from consts import Consts
from direction import Direction
from structs import Point


class Game:

	def __init__(self, width=Consts.DEFAULT_WINDOW_WIDTH, height=Consts.DEFAULT_WINDOW_HEIGHT):
		self.width = width
		self.height = height
		self.grid_width = width // Consts.TILE_SIZE
		self.grid_height = height // Consts.TILE_SIZE

		self.clock = pygame.time.Clock()
		self.display = pygame.display.set_mode(size=(width, height))
		pygame.display.set_caption("Snake Game")

		# Initializing snake:
		self.facing = Direction.get_random_direction()
		self.front, self.snake = self.create_snake()

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

	def loop(self):
		loop = True
		while loop:
			self.clock.tick(Consts.FPS)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					loop = False
			pygame.display.update()

		pygame.quit()


def run():
	pygame.init()
	Game().loop()
