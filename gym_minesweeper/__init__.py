from gym.envs.registration import register

register(
    id='gym_minesweeper/Minesweeper-v0',
    entry_point='gym_minesweeper.envs:MinesweeperEnv',
    max_episode_steps=16*16,
)