#!/usr/bin/env python3
import numpy as np
import gym
import tqdm


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """state action reward state action"""
    n = env.observation_space.n
    for _ in tqdm.tqdm(range(episodes)):
        state = env.reset()
        E = np.zeros((n, env.action_space.n))
        action = np.argmax(Q[state, :])
        for _ in range(max_steps):
            ss, r, done, _ = env.step(action)
            aa = np.argmax(Q[ss, :])
            delta = r + gamma * Q[ss, aa] - Q[state, action]
            E[state, action] += 1
            for s in range(n):
                for a in range(env.action_space.n):
                    Q[s, a] += alpha * delta * E[s, a]
                    E[s, a] *= gamma * lambtha
            if done:
                break
            state = ss
            action = aa
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(Q[state, :])
        if epsilon > min_epsilon:
            epsilon -= epsilon_decay

        # Print statements to observe computations
    return Q
