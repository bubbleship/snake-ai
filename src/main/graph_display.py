import json
import os

from matplotlib import pyplot, ticker

from consts import HistoryDataNames as HDN, Consts

pyplot.ion()


class History:
	scores_history: list[int] = []
	scores_average_history: list[float] = []
	highest_score_history: list[int] = []
	total_score: int = 0
	episodes_count: int = 0
	highest_score: int = 0

	@staticmethod
	def add_record(score: int):
		History.scores_history.append(score)
		History.total_score += score
		History.episodes_count += 1
		History.scores_average_history.append(History.total_score / History.episodes_count)

		if score >= History.highest_score:
			History.highest_score = score
			History.highest_score_history.append(History.episodes_count)

	@staticmethod
	def save():
		data = {
			HDN.SCORES_HISTORY: History.scores_history,
			HDN.SCORES_AVERAGE_HISTORY: History.scores_average_history,
			HDN.HIGHEST_SCORE_HISTORY: History.highest_score_history,
			HDN.TOTAL_SCORE: History.total_score,
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
		History.highest_score_history = data[HDN.HIGHEST_SCORE_HISTORY]
		History.total_score = data[HDN.TOTAL_SCORE]
		History.episodes_count = data[HDN.EPISODES_COUNT]

		History.highest_score = 0 if History.episodes_count == 0 else History.scores_history[
			History.highest_score_history[-1] - 1]

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
		axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
		plot_range = range(1, History.episodes_count + 1)
		axes.plot(plot_range, History.scores_history, color="green", marker="o", ms=5)
		axes.plot(plot_range, History.scores_average_history, color="blue", marker="o", ms=5)
		axes.plot(History.highest_score_history,
				  [History.scores_history[i - 1] for i in History.highest_score_history], color="orange", marker="o",
				  mfc="green", mec="orange", ls="--")
		axes.legend(["Score", "Average", "Highest"])
		axes.grid(True)
		axes.set_ylim(ymin=0)
		axes.set_xlim(xmin=0)
		if History.episodes_count != 0:
			axes.text(History.highest_score_history[-1] + 0.3, History.highest_score, str(History.highest_score))
			axes.text(len(History.scores_history) + 0.3, History.scores_history[-1], str(History.scores_history[-1]))
			axes.text(len(History.scores_average_history) + 0.3, History.scores_average_history[-1],
					  str(round(History.scores_average_history[-1], 2)))
		pyplot.show(block=True)
