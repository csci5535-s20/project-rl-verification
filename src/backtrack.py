import numpy as np
import random
from collections import deque
from keras.optimizers import Adam
import gym
from utils.utils import LossHistory, buildModel
from utils.visualize import VisualizeScores
import os

ENV_NAME = 'CartPole-v1'

class backtrackDQN:
  """
  Deep Q-learning network that implements backtracking to learn the control policy behind balancing the cartpole (or other envs).

  Args:
    num_states: int, the dimension of the state space or observation_space.
    num_actions: int, the dimension of the action space.
    epsilon: float, the epsilon-greedy value i.e exploration probability.
    gamma: float, discount factor
    decay_rate: float, the rate of decay for the epsilon value after each epoch.
    min_epsilon: float, the value which epsilon will approach as the training progresses.
    memory_size: int, size of replay buffer
    batch_size: int, batch size for the DQN model
    learning_rate: float, learning rate for the policy
  """
  def __init__(self, num_states, num_actions, epsilon=1.0, gamma=0.95, decay_rate=0.995,
               min_epsilon=0.01, memory_size=100000, batch_size=20, learning_rate=1e-3):
    self.epsilon = epsilon # exploration probability
    self.gamma = gamma # discount factor
    self.decay_rate = decay_rate # decaying rate for epsilon
    self.min_epsilon = min_epsilon # do not go below min_epsilon
    self.memory_size = memory_size # buffer size
    self.memory = deque(maxlen=memory_size) # list to store memory (replay buffer)
    self.batch_size = batch_size
    self.lr = learning_rate
    self.num_states = num_states # number of states comes from the observation space
    self.num_actions = num_actions # get n dimensions of the discrete action space

    # build the model
    self.model = buildModel(input_shape=self.num_states, output_shape=self.num_actions)
    print(self.model.summary())
    self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.lr))

  def chooseAction(self, state):
    """ Choose an action randomly as long as under epsilon. Else, take the argmax """
    if np.random.rand() < self.epsilon:
      # Choose a random action if under epsilon
      action = np.random.randint(self.num_actions)
    else:
      # Take action that has the maximum expected reward
      action = np.argmax(self.model.predict(state), axis=1)[0]
    return action

  def addToReplayBuffer(self, state, action, reward, new_state, terminal):
    """ Store the state, action, reward, next state and terminal status """
    self.memory.append((state, action, reward, new_state, terminal))

  def replayBuffer(self):
    """ Experience replay or replay buffer used to store state, action and info about the agent's parameters"""
    if len(self.memory) < self.batch_size:
      return
    # backtracking starts by sampling the memory
    batch = random.sample(self.memory, self.batch_size)
    for state, action, reward, next_state, terminal in batch:
      q_new = reward
      if not terminal:
        q_new = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
      q_values = self.model.predict(state)
      q_values[0][action] = q_new
      self.model.fit(state, q_values, verbose=0)
    self.epsilon *= self.decay_rate # decay the exploration rate
    self.epsilon = max(self.min_epsilon, self.epsilon)


def run(args):
  """Runs the trained model"""

  env = gym.make(ENV_NAME)
  backtrack_logger = VisualizeScores(ENV_NAME)
  observation_space = env.observation_space.shape[0]
  action_space = env.action_space.n
  dqn_backtrack = backtrackDQN(observation_space, action_space, learning_rate=args.lr, batch_size=args.batch_size)
  episode = 0
  for epoch in range(args.num_epochs):
    step = 0
    total_reward = 0
    episode += 1
    state = env.reset()
    state = np.reshape(state, (1, observation_space))
    while True:
      step += 1
      env.render() # visualize the environment
      action = dqn_backtrack.chooseAction(state)
      next_state, reward, terminal, info = env.step(action)
      total_reward += reward
      next_state = np.reshape(next_state, (1, observation_space))
      # add the new information to memory
      dqn_backtrack.addToReplayBuffer(state, action, total_reward, next_state, terminal)
      state = next_state
      if terminal:
        print('Episode: {}, exploration: {}, score:'\
              ' {}'.format(episode, np.round(dqn_backtrack.epsilon, 3), step))
        backtrack_logger.add_score(step, episode, output_path=os.path.join(args.output_dir,args.output_filename))
        break
      dqn_backtrack.replayBuffer()
