from enum import Enum


class Consts:
	DEFAULT_WINDOW_WIDTH = 640
	DEFAULT_WINDOW_HEIGHT = 640
	FPS_PLAYABLE = 15
	FPS_AGENT = 60
	TILE_SIZE = 20
	MAX_MEMORY = 100_000
	BATCH_SIZE = 1000
	LEARNING_RATE = 0.001
	MODEL_INPUT_LAYER_SIZE = 11
	MODEL_HIDDEN_LAYER_SIZE = 64
	MODEL_OUTPUT_LAYER_SIZE = 3
	MODEL_DIR_PATH = "model"
	MODEL_FILE_NAME = "model.pth"
	DATA_FILE_NAME = "data.json"
	AGENT_DATA_FILE_NAME = "agent_data.json"


class AgentDataNames:
	EPSILON = "epsilon"


class HistoryDataNames:
	SCORES_HISTORY = "scores_history"
	SCORES_AVERAGE_HISTORY = "scores_average_history"
	HIGHEST_SCORE_HISTORY = "highest_score_history"
	TOTAL_SCORE = "total_score"
	EPISODES_COUNT = "episodes_count"


class Colors:
	SCORE = (200, 0, 0)
	SNAKE_OUTLINE = (63, 186, 0)
	SNAKE_FILL = (86, 255, 0)
	BACKGROUND = (0, 0, 0)
	TEXT = (169, 183, 198)


class Action(Enum):
	FORWARD = 0
	LEFT = 1
	RIGHT = 2
