import tensorflow as tf
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def CNN():
  model = Sequential()
  # [(W−K+2P)/S] + 1
  # W is the input volume(),K is the Kernel size(5),P is the padding (0),S is the stride (2)
  model.add(Convolution2D(24,(5,5),(2,2),input_shape=(66,200,3),activation='elu')) #66*200*3
  model.add(Convolution2D(36,(5,5),(2,2),activation='elu')) #31*98*24
  model.add(Convolution2D(48,(5,5),(2,2),activation='elu')) #14*47*36
  model.add(Convolution2D(64,(3,3),(1,1),activation='elu')) #5*22*48
  model.add(Convolution2D(64,(3,3),(1,1),activation='elu')) #3*20*64
  #O/p size 1*18*64

  model.add(Flatten())
  model.add(Dense(100,activation='elu'))
  model.add(Dense(50,activation='elu'))
  model.add(Dense(10,activation='elu'))
  model.add(Dense(4))
  model.compile(optimizer=Adam(learning_rate=0.0001),loss='mse')

  return model 


