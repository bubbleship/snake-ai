from collections import deque

import pygame

from agent import Agent
from consts import Consts
from graph_display import History
from snake_game_agent import Game
from structs import Transition


def start():
	agent: Agent = Agent(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_OUTPUT_LAYER_SIZE)
	game = Game()
	memory = deque(maxlen=Consts.MAX_MEMORY)

	while game.is_running:  # Training loop.
		previous_state = game.get_state()  # Getting the game state before the action.

		action = agent.get_action(previous_state)  # Getting the action.
		reward, is_game_over, score = game.loop_iteration(action)  # Applying the action.

		next_state = game.get_state()  # Getting the game state after the action.

		memory.append(Transition(previous_state, action.value, reward, next_state, is_game_over))
		agent.train(memory)

		if is_game_over:
			game.reset()
			History.add_record(score)
			print("Game", History.episodes_count, "Score", score, "Highest", History.highest_score)

	pygame.quit()
	History.plot_history()


if __name__ == "__main__":
	start()
