import pygame

from consts import Consts


class Game:

	def __init__(self, width=Consts.DEFAULT_WINDOW_WIDTH, height=Consts.DEFAULT_WINDOW_HEIGHT):
		self.width = width
		self.height = height
		self.clock = pygame.time.Clock()
		self.display = pygame.display.set_mode(size=(width, height))
		pygame.display.set_caption("Snake Game")

	def loop(self):
		loop = True
		while loop:
			self.clock.tick(Consts.FPS)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					loop = False
			pygame.display.update()

		pygame.quit()
