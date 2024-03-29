from collections import deque

import pygame

from agent.agent import Agent
from agent.graph_display import History
from agent.snake_game_agent import Game
from commons.consts import Consts
from commons.structs import Transition


def start():
	agent: Agent = Agent()
	game = Game()
	memory: deque = deque(maxlen=Consts.MAX_MEMORY)

	agent.load()
	History.load()

	while game.is_running:  # Training loop.
		previous_state = game.get_state()  # Getting the game state before the action.

		action = agent.get_action(previous_state)  # Getting the action.
		reward, is_game_over, score = game.loop_iteration(action)  # Applying the action.

		next_state = game.get_state()  # Getting the game state after the action.

		memory.append(Transition(previous_state, action.value, reward, next_state, is_game_over))
		agent.train(memory)

		if is_game_over:
			agent.decay_epsilon()
			game.reset()
			History.add_record(score)
			print("Game", History.episodes_count, "Score", score, "Highest", History.highest_score)

	pygame.quit()
	History.plot_history()

	agent.save()
	History.save()


if __name__ == "__main__":
	start()
