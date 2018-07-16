import numpy as np
import sys
sys.path.append("game/")

import skimage
from skimage import transform, color, exposure

from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *

import tensorflow as tf

import pygame
import os,sys
work_space = os.path.dirname(os.path.abspath(__file__))
sys.path.append(work_space+"/game/")
import wrapped_flappy_bird as game

import time
import math

GAMMA = 0.99                #discount value
BETA = 0.01                 #regularisation coefficient
IMAGE_ROWS = 85
IMAGE_COLS = 84
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4
EPISODE = 0
THREADS = 16
t_max = 5
const = 1e-5
T = 0

# reward list of a episode
episode_r = []
# state list of a episode
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
# predicted action list of a episode
episode_output = []
# predicted reward list of a episode
episode_critic = []

ACTIONS = 2
a_t = np.zeros(ACTIONS)

# #loss function for policy output
# def logloss(y_true, y_pred):     #policy loss
# 	return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + const), axis=-1)
# 	# BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) * K.log(1-y_pred + const))   #regularisation term
#
# #loss function for critic output
# def sumofsquares(y_true, y_pred):        #critic loss
# 	return K.sum(K.square(y_pred - y_true), axis=-1)

def logloss(y_true,y_pred):
	return -sum(log(y_true*y_pred + (1-y_true)*(1-y_pred)),axis=-1)

def sumofsquares(y_true, y_pred):
	return sum(square(y_pred-y_true),axis=-1)

#function buildmodel() to define the structure of the neural network in use
def buildmodel():
	print("Model building begins")

	model = Sequential()
	# keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S = Input(shape = (IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name = 'Input')
	h0 = Convolution2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(S)
	h1 = Convolution2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform')(h0)
	h2 = Flatten()(h1)
	h3 = Dense(256, activation = 'relu', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h2)
	P = Dense(1, name = 'o_P', activation = 'sigmoid', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)
	V = Dense(1, name = 'o_V', kernel_initializer = 'random_uniform', bias_initializer = 'random_uniform') (h3)

	model = Model(inputs = S, outputs = [P,V])
	rms = RMSprop(lr = LEARNING_RATE, rho = 0.99, epsilon = 0.1)
	model.compile(loss = {'o_P': logloss, 'o_V': sumofsquares}, loss_weights = {'o_P': 1., 'o_V' : 0.5}, optimizer = rms)
	return model





