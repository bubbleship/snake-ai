from collections import deque
from typing import Any

from agent import Agent
from consts import Consts
from snake_game_agent import Game
from structs import Transition


def start():
	scores_history: list[Any] = []
	scores_average_history: list[float] = []
	total_score: int = 0
	highest_score: int = 0
	games_count: int = 0
	agent: Agent = Agent(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_OUTPUT_LAYER_SIZE)
	game = Game()
	memory = deque(maxlen=Consts.MAX_MEMORY)

	while True:  # Training loop.
		previous_state = game.get_state()  # Getting the game state before the action.

		action = agent.get_action(previous_state)  # Getting the action.
		reward, is_game_over, score = game.loop_iteration(action)  # Applying the action.

		next_state = game.get_state()  # Getting the game state after the action.

		memory.append(Transition(previous_state, action.value, reward, next_state, is_game_over))
		agent.train(memory)

		if is_game_over:
			game.reset()
			games_count += 1

			if score > highest_score:
				highest_score = score

			print("Game", games_count, "Score", score, "Highest", highest_score)

			scores_history.append(score)
			total_score += score
			scores_average = total_score / games_count
			scores_average_history.append(scores_average)


if __name__ == "__main__":
	start()
