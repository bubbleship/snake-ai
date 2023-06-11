import matplotlib.pyplot as pyplot

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
