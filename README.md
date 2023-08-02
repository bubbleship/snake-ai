# snake-ai

![Licence](https://img.shields.io/github/license/bubbleship/snake-ai)

### Description

This project aims to develop an AI agent that can learn to play the classic game Snake using deep
reinforcement learning. The application includes a graphical interface where the interactions of
the agent with the game environment can be seen. Closing the graphical interface will show a
graph summary of the agent's performance throughout the session.

### How it Works

The environment is a grid where each tile is either empty space, score, or a part of the snake.
The goal of the agent is to collide with as many score tiles as possible while avoiding
collision with snake tiles (itself) and the edges.
<br>
With every step of the game the agent is given the following insights about the environment:

1. Whether there is a collision risk to the left, right and in front of the head of the snake.
2. The direction where the snake is facing.
3. The location of the score relative to the head of the snake.

The agent begins by taking random actions to explore the environment and, as it gathers enough
experience, it slowly turns to relying on a Deep Q-Learning model to take more calculated actions.

### Limitations

As the only information the agent receives about obstacles in the environment consists of three tiles
around the head of the snake the agent is more likely to spiral into itself.

### Requirements

Python 3.10.0 and above.
<br>
This project may work with earlier versions, but it is not verified.

**Libraries**

* PyTorch and NumPy: for the implementation of the agent
* Pygame: for the implementation of the environment
* Matplotlib: for the graph summary of the agent's performance

