import os
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium import Wrapper
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# Constants and Configuration
ENV_NAME = "ALE/Breakout-v5"
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

class AtariProcessor(Processor):
    """ Preprocess observations and rewards"""
    def process_observation(self, observation):
        """process env frames to 84x84 grayscale image"""
        # Preprocess observation (resize, grayscale)
        if isinstance(observation, tuple):
            observation = observation[0]  # Extract screen part
        observation = np.array(observation)
        assert observation.ndim == 3, "Observation does not have 3 dimensions"
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """normalize state batch to 0-1 range"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """clip reward to -1, 1 range"""
        return np.clip(reward, -1., 1.)

class CustomWrapper(Wrapper):
    """class for modifying environment behavior"""
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """adjust environment step to work with keras-rl"""
        observation, reward, done, _, info = self.env.step(action)
        if isinstance(info, dict):
            # Modify info dictionary if necessary
            info.pop('episode_frame_number', None)
            info.pop('frame_number', None)
        return observation, reward, done, info

def setup_environment():
    """build and return custom environment with wrapper and seed set to 123"""
    env = gym.make(ENV_NAME)
    env = CustomWrapper(env)
    np.random.seed(123)
    env.seed(123)
    return env

# Define DQN model
def build_model(input_shape, nb_actions):
    """build and return DQN model with input shape and number of actions as parameters"""
    inp = Input(shape=input_shape)
    X = Permute((2, 3, 1))(inp)
    X = Convolution2D(32, (8, 8), strides=(4, 4))(X)
    X = Activation('relu')(X)
    X = Convolution2D(64, (4, 4), strides=(2, 2))(X)
    X = Activation('relu')(X)
    X = Convolution2D(64, (3, 3), strides=(2, 2))(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Activation('relu')(X)
    X = Dense(nb_actions)(X)
    X = Activation('linear')(X)
    model = Model(inputs=inp, outputs=X)
    return model

def build_agent(model, nb_actions, memory, processor):
    """setup and return DQN agent with model, number of actions, memory, and processor as parameters"""
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])
    return dqn

def train_agent(env, dqn, callbacks, nb_steps, log_interval):
    """trainer function for DQN agent with environment, agent, callbacks, number of steps, and log interval as parameters"""
    dqn.fit(env, callbacks=callbacks, nb_steps=nb_steps, log_interval=log_interval)

# Save and load DQN agent weights
def save_load_weights(dqn, weights_filename):
    """save and load weights for DQN agent with agent and weights filename as parameters"""
    dqn.save_weights(weights_filename, overwrite=True)
    dqn.load_weights(weights_filename)

if __name__ == "__main__":
    log_dir = 'dqn_ALE'

    # Create the directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Now you can use the directory path for saving log files
    log_filename = os.path.join(log_dir, 'Breakout-v5_log.json')
    env = setup_environment()
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    nb_actions = env.action_space.n
    model = build_model(input_shape, nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()

    # Build DQN agent
    dqn = build_agent(model, nb_actions, memory, processor)

    # Callbacks setup
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    # Training
    train_agent(env, dqn, callbacks, nb_steps=1750000, log_interval=5000)

    # Save and load weights
    save_load_weights(dqn, weights_filename)
