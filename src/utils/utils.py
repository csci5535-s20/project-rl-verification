import keras
from keras.models import Sequential
from keras.layers import Dense

class LossHistory(keras.callbacks.Callback):
  """ Custom Keras callback to print and save loss after every batch. """
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))


def buildModel(input_shape, output_shape):
  """ Builds a neural network that approximates a set of actions for each input state. """
  model = Sequential()
  model.add(Dense(36, input_shape=(input_shape,), activation='relu'))
  model.add(Dense(36, activation='relu'))
  model.add(Dense(output_shape, activation='linear'))
  return model
