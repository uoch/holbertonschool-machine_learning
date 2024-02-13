#!/usr/bin/env python3
"""Q-learning module"""
import gym
import numpy as np


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """loads the pre-made FrozenLakeEnv environment from OpenAI’s gym"""
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
