#!/usr/bin/env python3
"""Q-learning module"""
import gym
import numpy as np


def play(env, Q, max_steps=100):
    """play an episode"""
    state = env.reset()
    tot = 0
    print(env.render())
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        print(env.render(mode="ansi"))
        if done:
            print(env.render(mode="ansi"))
            break
        state = new_state
        tot += reward
    return tot
