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

def train(env, nb_episodes, alpha=0.000045, gamma=0.98 , show_result = False):
    """training function"""
    # initialize the weight
    weight = he_init((env.observation_space.shape[0], env.action_space.n))

    scores = []
    for ep in tqdm.tqdm(range(nb_episodes)):
        state, _ = env.reset()
        episode = []
        grads = []
        episode_score = 0
        while True:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            grads.append(grad)
            episode_score += reward

            state = next_state
            if done:
                break
        if show_result and (ep+1) % 1000 == 0:
            env.render()
            st ,_= env.reset()
            img = env.render()
            imgs = []
            while True:
                action, _ = policy_gradient(st, weight)
                state, reward, done, _, _ = env.step(action)
                img = env.render()
                imgs.append(img)
                if done:
                    break
            gif_path = f'gif/cartpole{ep+1}.gif'
            imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(imgs) ], fps=1000)
            display.display(display.Image(filename=gif_path))
            time.sleep(1.5)  # Display the GIF for 2 seconds
            display.clear_output(wait=True)  # Clear the output after displaying the GIF
            env.close()
        scores.append(episode_score)
        for i in range(len(episode)):
            Gt = sum([gamma**j * episode[j][2]
                     for j in range(i, len(episode))])
            step, action, _ = episode[i]
            action_probs = policy(step, weight)
            Ln = np.log(action_probs[action])
            weight += alpha * gamma**i * Gt * Ln * grads[i]
    return scores 
