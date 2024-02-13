#!/usr/bin/env python3
"""Q-learning module"""
import gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def bellman_equation(Q, state, action, reward, next_state, alpha, gamma):
    """performs the Bellman equation"""
    Q[state, action] = (Q[state, action] * (1 - alpha) +
                        (reward + gamma * np.max(Q[next_state, :])) * alpha)
    return Q


def train(env, Q,
          episodes=5000,
          max_steps=100,
          alpha=0.1, gamma=0.99,
          epsilon=1,
          min_epsilon=0.1,
          epsilon_decay=0.05):
    """teach agent how to play FrozenLake"""
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
                reward = -1
            Q = bellman_equation(Q, state, action, reward,
                                 new_state, alpha, gamma)
            total_rewards += reward
            state = new_state
            if done:
                break

        epsilon = (min_epsilon + (epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * ep))
        rewards.append(total_rewards)
    return Q, rewards
