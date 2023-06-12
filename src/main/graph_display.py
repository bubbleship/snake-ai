import json
import os

import matplotlib.pyplot as pyplot

from consts import HistoryDataNames as HDN, Consts

pyplot.ion()


class History:
	scores_history: list[int] = []
	scores_average_history: list[float] = []
	total_score: int = 0
	highest_score: int = 0
	episodes_count: int = 0

	@staticmethod
	def add_record(score: int):
		if score > History.highest_score:
			History.highest_score = score

		History.scores_history.append(score)
		History.total_score += score
		History.episodes_count += 1
		History.scores_average_history.append(History.total_score / History.episodes_count)

	@staticmethod
	def save():
		data = {
			HDN.SCORES_HISTORY: History.scores_history,
			HDN.SCORES_AVERAGE_HISTORY: History.scores_average_history,
			HDN.TOTAL_SCORE: History.total_score,
			HDN.HIGHEST_SCORE: History.highest_score,
			HDN.EPISODES_COUNT: History.episodes_count
		}

		with open(os.path.join(Consts.MODEL_DIR_PATH, Consts.DATA_FILE_NAME), "w") as file:
			json.dump(data, file, indent=2)

	@staticmethod
	def load():
		path = os.path.join(Consts.MODEL_DIR_PATH, Consts.DATA_FILE_NAME)
		if not os.path.exists(path):
			return

		with open(path, "r") as file:
			data = json.load(file)

		History.scores_history = data[HDN.SCORES_HISTORY]
		History.scores_average_history = data[HDN.SCORES_AVERAGE_HISTORY]
		History.total_score = data[HDN.TOTAL_SCORE]
		History.highest_score = data[HDN.HIGHEST_SCORE]
		History.episodes_count = data[HDN.EPISODES_COUNT]

	@staticmethod
	def plot_history():
		History._plot_history()

	@staticmethod
	def _plot_history():
		figure, axes = pyplot.subplots(nrows=1, ncols=1)

		figure.canvas.manager.set_window_title("Training History")
		axes.set_title("Training History")
		axes.set_xlabel("Game #")
		axes.set_ylabel("Score")
		axes.set_xticks(range(History.episodes_count), range(1, History.episodes_count + 1))
		axes.plot(History.scores_history, color="green", marker="o")
		axes.plot(History.scores_average_history, color="blue", marker="o")
		axes.legend(["Score", "Average"])
		axes.grid(True)
		axes.set_ylim(ymin=0)
		axes.set_xlim(xmin=0)
		if History.episodes_count != 0:
			axes.text(len(History.scores_history) - 1, History.scores_history[-1], str(History.scores_history[-1]))
			axes.text(len(History.scores_average_history) - 1, History.scores_average_history[-1],
					  str(History.scores_average_history[-1]))
		pyplot.show(block=True)
