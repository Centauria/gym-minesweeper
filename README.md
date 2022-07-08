# Gym Minesweeper
**Gym Minesweeper is an environment for OpenAI Gym simulating minesweeper game.**

![Minesweeper Solver](https://jeffreyyao.github.io/images/minesweeper_solver.gif)

---

## Installation

```bash
cd gym-minesweeper
pip install -e .
```

## Running

```python
import gym
import gym_minesweeper

env = gym.make("Minesweeper-v0") # 16x16 map with 40 mines
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

env.close()
```