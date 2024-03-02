#!/usr/bin/env python3
import numpy as np
import gym
import tqdm

def reward_discount(rewards, gamma):
    """returns the reward discount"""
    return np.sum([rewards[i] * (gamma ** i) for i in range(len(rewards))])

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """performs the Monte Carlo algorithm"""
    n = env.observation_space.n
    returns = {s: [] for s in range(n)}
    # run the episodes
    for _ in tqdm.tqdm(range(episodes)):
        state = env.reset()
        episode = []
        # collect the rewards
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = new_state
        # compute monte carlo
        episode = list(set([tuple(i) for i in episode]))
        G = 0
        # run backwards through the episode
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            # G is sum of weighted rewards
            G = reward + gamma * G
            if not (state, action) in episode[0:i]:
                returns[state].append(G)
                V[state] = V[state] + alpha * (reward_discount(returns[state], gamma) - V[state])
    return V