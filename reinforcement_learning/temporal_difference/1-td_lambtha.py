#!/usr/bin/env python3
import numpy as np
import gym
import tqdm

def G_lambda_step(step, rewards, gamma, lambtha):
    """returns: the discounted rewards"""
    G = 0
    for i in range(step, len(rewards)):
        G += rewards[i] * (gamma ** (i - step))
    return G * (1 - lambtha)


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """temporal difference Lambda algorithm"""
    n = env.observation_space.n
    for _ in tqdm.tqdm(range(episodes)):
        state = env.reset()
        E = np.zeros(n)
        rewards = []
        for step in range(max_steps):
            action = policy(state)
            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            G = G_lambda_step(step, rewards, gamma, lambtha)
            E = gamma * lambtha * E
            E[state] = E[state] + 1
            V = V + alpha * (G - V[state]) * E
            if done:
                break
            state = new_state
    return V
