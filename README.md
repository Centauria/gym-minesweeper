# Gym Minesweeper
Gym Minesweeper is an environment for simulating minesweeper game with custom sizes and number of mines. Built on top of pygame.

![Minesweeper Solver](https://jeffreyyao.github.io/images/minesweeper_solver.gif)

## Installation

```bash
cd gym-minesweeper
pip install -e .
```

## Running

```python
import gym

env = gym.make("Minesweeper-v0") # 16x16 map with 40 mines
env.init()
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
```