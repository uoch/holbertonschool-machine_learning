import tqdm
import imageio
import IPython.display as display
import time
from policy_gradient import policy, policy_gradient
import numpy as np


def xavier_init(shape):
    """Xavier initialization for weights"""
    fan_in = shape[0]
    fan_out = shape[1]
    stddev = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, stddev, shape)


def he_init(shape):
    """He initialization for weights"""
    fan_in = shape[0]
    stddev = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, stddev, shape)


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, epsilon=0.1, show_result=False, initial=he_init):
    """Training function"""
    # Initialize the weight
    weight = initial((env.observation_space.shape[0], env.action_space.n))
    mean_state = np.zeros(env.observation_space.shape[0])
    std_state = np.ones(env.observation_space.shape[0])
    scores = []

    for ep in tqdm.tqdm(range(nb_episodes)):
        state ,_= env.reset()
        episode = []
        grads = []
        episode_score = 0

        while True:
            n_state = (state - mean_state) / std_state

            # Epsilon-greedy exploration
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
                if np.isnan(action).any():
                    action = np.ones_like(action) / len(action)
                _, grad = policy_gradient(n_state, weight)
            else:
                action, grad = policy_gradient(n_state, weight)
                if np.isnan(action).any():
                    action = np.ones_like(action) / len(action)
                action_probs = policy(state, weight)
                if np.isnan(action_probs).any():
                    action = np.ones_like(action_probs) / len(action_probs)

            next_state, reward, done, _,_ = env.step(action)
            episode.append((state, action, reward))
            grads.append(grad)
            episode_score += reward

            state = next_state
            if done:
                break

        states = np.array([step[0] for step in episode])
        mean_state = np.mean(states, axis=0)
        std_state = np.std(states, axis=0)

        if show_result and (ep + 1) % 1000 == 0:
            env.render()
            st, _= env.reset()
            img = env.render()
            imgs = []

            while True:
                action, _ = policy_gradient(st, weight)
                state, reward, done, _,_ = env.step(action)
                img = env.render()
                imgs.append(img)
                if done:
                    break

            gif_path = f'gif/cartpole{ep + 1}.gif'
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(imgs)], fps=1000)
            display.display(display.Image(filename=gif_path))
            time.sleep(1.5)  # Display the GIF for 2 seconds
            display.clear_output(wait=True)  # Clear the output after displaying the GIF
            env.close()

        scores.append(episode_score)

        Gt = 0
        for i in range(len(episode)):
            Gt = sum([gamma ** (j - i) * episode[j][2] for j in range(i, len(episode))])
            step, action, _ = episode[i]
            action_probs = policy(step, weight)
            if np.isnan(action_probs).any():
                action = np.ones_like(action_probs) / len(action_probs)
            Ln = np.log(action_probs[action])
            weight += alpha * Gt * Ln * grads[i]

    return scores, weight
