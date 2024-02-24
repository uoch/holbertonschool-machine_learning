import gym
import gymnasium as gym
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Constants and Configuration
ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

# Load the environment
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

# Load the trained DQN model
model = load_model('path_to_your_model.h5')

# Preprocessing function
def preprocess_observation(observation):
    """ Preprocess observation """
    # Preprocess observation (resize, grayscale)
    observation = np.array(observation)
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')
    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    processed_observation = np.expand_dims(processed_observation, axis=0)
    return processed_observation.astype('uint8')

# Play loop
observation = env.reset()
done = False
while not done:
    # Preprocess the observation
    processed_observation = preprocess_observation(observation)

    action = np.argmax(model.predict(processed_observation))

    observation, reward, done, _ = env.step(action)

    env.render()

env.close()
