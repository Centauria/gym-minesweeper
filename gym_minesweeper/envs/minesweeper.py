import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.signal import convolve2d
import pygame
import pygame.freetype


class MinesweeperEnv(gym.Env):
    # a minesweeper environment implemented in openai gym style
    # STATE
    #   given the map of height x width, numbers in each block of map:
    #   1. -1 denotes unopened block
    #   2. 0-8 denotes the number of mines in the surrounding 8 blocks of the block
    # MAP
    #   the map is unobservable to the agent where 0 is non-mine, 1 has a mine
    # ACTIONS
    #   action is Discrete(height*width) representing an attempt to open at the block's
    #   index when the map is flattened.
    # RESET
    #   be sure to call the reset function before using any other function in the class
    # STEP(ACTION)
    #   returns a four tuple: next_state, reward, done, _
    # RENDER
    #   renders the current state using pygame

    def __init__(self, height=16, width=16, num_mines=40, prevent_first_bomb=True):
        self.observation_space = spaces.Box(-1, 8, shape=(height, width), dtype=int)
        self.action_space = spaces.MultiDiscrete([height, width])

        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.prevent_first_bomb = prevent_first_bomb

        self.win_reward = 50
        self.fail_reward = -10
        self.map = None
        self.state = None
        self.surroundings = None
        self.step_counter = 0
        self.step_counter_max = (height * width - num_mines) * 2
        self.core = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)

        self.block_size = 25
        self.window_height = self.block_size * height
        self.window_width = self.block_size * width

        self.screen = None

    def generate_mines(self, start_position=None):
        samples = self.height * self.width
        self.map = np.zeros(samples, dtype=np.int8)
        if start_position is None:
            self.map[np.random.choice(samples, self.num_mines, replace=False)] = 1
        else:
            self.map[np.random.choice(samples - 1, self.num_mines, replace=False)] = 1
            y, x = start_position
            index = y * self.width + x
            self.map[-1] = self.map[index]
            self.map[index] = 0
        self.map = self.map.reshape(self.height, self.width)
        self.surroundings = convolve2d(self.map, self.core, mode="same")

    def reset(self, seed=None, return_info=False, options=None):
        if not self.prevent_first_bomb:
            self.generate_mines()
        self.step_counter = 0
        self.state = -np.ones((self.height, self.width), dtype=np.int8)
        return self.state

    def get_num_opened(self):
        return (self.state >= 0).astype(int).sum()

    def update_state(self, y, x):
        self.state[y, x] = self.surroundings[y, x]
        if self.state[y, x] == 0:
            for j in range(max(0, y - 1), min(self.height, y + 2)):
                for i in range(max(0, x - 1), min(self.width, x + 2)):
                    if (not (i == x and j == y)) and self.state[i, j] == -1:
                        self.update_state(i, j)

    def step(self, action):
        if len(action) != 2:
            raise ValueError
        y, x = action[0], action[1]
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            raise ValueError
        if self.map is None:
            self.generate_mines(action)
        info = self._get_info()
        if self.step_counter == self.step_counter_max:
            return self.state, 0, True, info
        else:
            self.step_counter += 1
        if self.map[y, x]:
            return self.state, self.fail_reward, True, info
        else:
            num_opened = self.get_num_opened()
            if self.state[y, x] != -1:
                return self.state, 0, False, info
            self.update_state(y, x)
            new_num_opened = self.get_num_opened()
            if new_num_opened == self.height * self.width - self.num_mines:
                return self.state, self.win_reward, True, info
            return self.state, new_num_opened - num_opened, False, info

    def draw_grid(self):
        for y in range(0, self.window_width, self.block_size):
            for x in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(y, x, self.block_size, self.block_size)
                num = int(self.state[x // self.block_size, y // self.block_size])
                if num == -1:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
                else:
                    color = (250, 250 - num * 30, 250 - num * 30)
                    pygame.draw.rect(self.screen, color, rect)
                    text = self.font.get_rect(str(num))
                    text.center = rect.center
                    self.font.render_to(self.screen, text.topleft, str(num), (0, 0, 0))
        pygame.display.update()

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.font = pygame.freetype.SysFont(pygame.font.get_default_font(), 13)
        self.screen.fill((0, 0, 0))
        self.draw_grid()

    def _get_info(self):
        return {"map": self.map}

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
