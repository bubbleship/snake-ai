import random
from collections import deque

import torch
from numpy import ndarray

from consts import Consts
from graph_display import plot
from model import LinearQNet, QTrainer, device
from snake_game_agent import Game


class Agent:

	def __init__(self):
		self.games_count = 0
		self.epsilon = 0
		self.gamma = 0.9  # Discount rate
		self.memory = deque(maxlen=Consts.MAX_MEMORY)
		self.model = LinearQNet(Consts.MODEL_INPUT_LAYER_SIZE, Consts.MODEL_HIDDEN_LAYER_SIZE,
				Consts.MODEL_OUTPUT_LAYER_SIZE)
		self.trainer = QTrainer(self.model, Consts.LEARNING_RATE, self.gamma)

	def remember(self, previous_state: ndarray, action: list[3], reward: int, next_state: ndarray,
			is_game_over: bool) -> None:
		self.memory.append((previous_state, action, reward, next_state, is_game_over))

	def train_long_term_memory(self):
		if len(self.memory) > Consts.BATCH_SIZE:
			sample = random.sample(self.memory, Consts.BATCH_SIZE)
		else:
			sample = self.memory

		previous_states, actions, rewards, next_states, is_game_overs = zip(*sample)
		self.trainer.train_step(previous_states, actions, rewards, next_states, is_game_overs)

	def train_short_term_memory(self, previous_state, action, reward, next_state, is_game_over):
		self.trainer.train_step(previous_state, action, reward, next_state, is_game_over)

	def get_action(self, state: ndarray) -> list[3]:
		self.epsilon = 80 - self.games_count
		action = [0, 0, 0]

		if random.randint(0, 200) < self.epsilon:
			action_type = random.randint(0, len(action) - 1)
			action[action_type] = 1
		else:
			state_tensor = torch.tensor(state, dtype=torch.float).to(device)
			prediction = self.model(state_tensor)
			action_type = torch.argmax(prediction).item()
			action[action_type] = 1

		return action


def train():
	plot_scores = []
	plot_mean_scores = []
	total_score = 0
	highest_score = 0
	agent = Agent()
	game = Game()

	while True:
		previous_state = game.get_state()
		action = agent.get_action(previous_state)

		reward, is_game_over, score = game.loop_iteration(action)
		next_state = game.get_state()

		agent.train_short_term_memory(previous_state, action, reward, next_state, is_game_over)
		agent.remember(previous_state, action, reward, next_state, is_game_over)

		if is_game_over:
			game.reset()
			agent.games_count += 1
			agent.train_long_term_memory()

			highest_score = max(highest_score, score)

			if score > highest_score:
				highest_score = score
				agent.model.save()

			print("Game", agent.games_count, "Score", score, "Highest", highest_score)

			plot_scores.append(score)
			total_score += score
			score_average = total_score / agent.games_count
			plot_mean_scores.append(score_average)
			plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
	train()
