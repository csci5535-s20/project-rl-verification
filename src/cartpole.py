import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from utils.visualize import VisualizeScores
import argparse
import os
from utils.utils import buildModel
import backtrack

ENV_NAME = "CartPole-v1"

class baselineDQN:
  """ Deep Q Learning Network that learns the control policy to balance cartpole

  Args:
    observation_space: int, the dimension of the state space or observation_space.
    action_space: int, the dimension of the action space.
    batch_size: int, batch size for the DQN model
    lr: float, learning rate for the policy
    max_exploration: float, the epsilon-greedy value i.e exploration probability.
    min_exploration: float, the value which epsilon will approach as the training progresses.
    gamma: float, discount factor
    decay_factor: float, the rate of decay for the epsilon value after each epoch.
    memory_size: int, size of replay buffer
  """

  def __init__(self, observation_space, action_space,
               lr, batch_size,
               gamma=0.95, memory_size=100000,
               max_exploration=1.0, min_exploration=0.01, decay_factor=0.995):

    self.exploration_rate = max_exploration
    self.min_exploration_rate = min_exploration
    self.decay_factor = decay_factor
    self.gamma = gamma
    self.batch_size = batch_size
    self.observation_space = observation_space
    self.action_space = action_space
    self.memory = deque(maxlen=memory_size)

    # simple dense model
    self.model = buildModel(self.observation_space, self.action_space)
    self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))

  def addToMemory(self, state, action, reward, next_state, terminal):
    self.memory.append((state, action, reward, next_state, terminal))

  def act(self, state):
    if np.random.rand() < self.exploration_rate:
      return random.randrange(self.action_space)
    q_values = self.model.predict(state)
    return np.argmax(q_values[0])

  def experienceReplay(self):
    """ Experience replay or replay buffer used to store state, action and info about the agent's parameters"""
    if len(self.memory) < self.batch_size:
      return

    batch = random.sample(self.memory, self.batch_size)
    for state, action, reward, next_state, terminal in batch:
      q_update = reward
      if not terminal:
        q_update = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
      q_values = self.model.predict(state)
      q_values[0][action] = q_update
      self.model.fit(state, q_values, verbose=0)
    self.exploration_rate *= self.decay_factor
    self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate)


def cartpole(args):
  env = gym.make(ENV_NAME)
  logger = VisualizeScores(ENV_NAME)
  observation_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  dqn = baselineDQN(observation_space, action_space, lr=args.lr, batch_size=args.batch_size)
  episode = 0
  while True:
    # print('Starting new run ...\n')
    episode += 1
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    step = 0
    while True:
      step += 1
      env.render()
      action = dqn.act(state)

      next_state, reward, terminal, info = env.step(action)
      # If the simulation has not terminated then it gets a positive reward.
      # If it has terminated, i.e. the pole has fallen over/fail criteria met then it gets a negative reward
      reward = reward if not terminal else -reward
      next_state = np.reshape(next_state, [1, observation_space])

      # Add to memory info about what state you were in, what action you took,
      # irrespective of whether that was rewarding and what the next state was and then whether it terminated or not.
      dqn.addToMemory(state, action, reward, next_state, terminal)
      state = next_state # previous step influences current step
      if terminal:
        print('Episode: {}, exploration: {}, score:'\
              ' {}'.format(episode, np.round(dqn.exploration_rate,3), step))
        logger.add_score(step, episode, output_path=os.path.join(args.output_dir,args.output_filename))
        break
      dqn.experienceReplay()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=20, type=int, help='Batch size to be used for the model')
  parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for the model')
  parser.add_argument('--output_dir', default='scores/', type=str, help='Directory where the figures will be saved')
  parser.add_argument('--num_epochs', default=500, type=int, help='Number of epochs to train the DQN model')
  parser.add_argument('--output_filename', default='cartpole', type=str, help='Filename for the saved figures')
  parser.add_argument('--use_baseline', action='store_false', dest='model_type',
                      help='Flag, set to True by default to use baseline model')
  parser.add_argument('--use_backtracking', action='store_true', dest='model_type',
                      help='Flag, set to False by default to use baseline model')
  parser.set_defaults(model_type=False)
  args = parser.parse_args()

  if args.model_type:
    print('Using baseline model')
    cartpole(args)
  else:
    print('Using backtracking model')
    backtrack.run(args)
