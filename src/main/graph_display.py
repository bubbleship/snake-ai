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
		pyplot.gcf()
		pyplot.clf()
		pyplot.title('Training history')
		pyplot.xlabel('Number of Games')
		pyplot.ylabel('Score')
		pyplot.plot(History.scores_history)
		pyplot.plot(History.scores_average_history)
		pyplot.ylim(ymin=0)
		pyplot.text(len(History.scores_history) - 1, History.scores_history[-1], str(History.scores_history[-1]))
		pyplot.text(len(History.scores_average_history) - 1, History.scores_average_history[-1],
					str(History.scores_average_history[-1]))
		pyplot.show(block=True)
		pyplot.pause(.1)
