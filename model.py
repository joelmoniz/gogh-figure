import sys
import os
import time
import string
import random
import collections
import cPickle as pickle
import gzip
import ast

import numpy as np
import theano
import theano.tensor as T
import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import FlattenLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import SliceLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, rectify, elu, sigmoid, tanh, softplus
from lasagne.layers import batch_norm, BatchNormLayer

from utils import *

class Network(object):

	LOSS_NET_VERSION = 0.1

	MODEL_PATH = './data/model/'
	LOSS_NET_MODEL_FILE_NAME = "vgg16_loss_net.npz"
	LOSS_NET_MODEL_SIZE = 58863490
	LOSS_NET_DOWNLOAD_LINK = "TODO" + str(LOSS_NET_VERSION) + "TODO" + LOSS_NET_MODEL_FILE_NAME
	LOSS_NET_MODEL_FILE_PATH = MODEL_PATH + LOSS_NET_MODEL_FILE_NAME

	def __init__(self, **kwargs):
		self.network = {}
		self.network['loss_net'] = {}
		self.network['_net'] = {}

	def setup_loss_net(self, input_layer):
		"""
		Create a network of convolution layers based on the VGG16 architecture from the paper:
		"Very Deep Convolutional Networks for Large-Scale Image Recognition"

		Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
		License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

		Based on code in the Lasagne Recipes repository: https://github.com/Lasagne/Recipes
		"""
		loss_net = self.network['loss_net']
		loss_net['input'] = input_layer
		loss_net['conv1_1'] = ConvLayer(loss_net['input'], 64, 3, pad=1, flip_filters=False)
		loss_net['conv1_2'] = ConvLayer(loss_net['conv1_1'], 64, 3, pad=1, flip_filters=False)
		loss_net['pool1'] = PoolLayer(loss_net['conv1_2'], 2)
		loss_net['conv2_1'] = ConvLayer(loss_net['pool1'], 128, 3, pad=1, flip_filters=False)
		loss_net['conv2_2'] = ConvLayer(loss_net['conv2_1'], 128, 3, pad=1, flip_filters=False)
		loss_net['pool2'] = PoolLayer(loss_net['conv2_2'], 2)
		loss_net['conv3_1'] = ConvLayer(loss_net['pool2'], 256, 3, pad=1, flip_filters=False)
		loss_net['conv3_2'] = ConvLayer(loss_net['conv3_1'], 256, 3, pad=1, flip_filters=False)
		loss_net['conv3_3'] = ConvLayer(loss_net['conv3_2'], 256, 3, pad=1, flip_filters=False)
		loss_net['pool3'] = PoolLayer(loss_net['conv3_3'], 2)
		loss_net['conv4_1'] = ConvLayer(loss_net['pool3'], 512, 3, pad=1, flip_filters=False)
		loss_net['conv4_2'] = ConvLayer(loss_net['conv4_1'], 512, 3, pad=1, flip_filters=False)
		loss_net['conv4_3'] = ConvLayer(loss_net['conv4_2'], 512, 3, pad=1, flip_filters=False)
		loss_net['pool4'] = PoolLayer(loss_net['conv4_3'], 2)
		loss_net['conv5_1'] = ConvLayer(loss_net['pool4'], 512, 3, pad=1, flip_filters=False)
		loss_net['conv5_2'] = ConvLayer(loss_net['conv5_1'], 512, 3, pad=1, flip_filters=False)
		loss_net['conv5_3'] = ConvLayer(loss_net['conv5_2'], 512, 3, pad=1, flip_filters=False)

	def load_loss_net_weights(self):
		download_if_not_exists(LOSS_NET_MODEL_FILE_PATH, LOSS_NET_DOWNLOAD_LINK, \
			"Downloading the Loss Network's weights", LOSS_NET_MODEL_SIZE)
		load_params(self.network['loss_net'], LOSS_NET_MODEL_FILE_PATH)

	def setup_transform_net(input_var=None):
		network = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
		network = style_conv_block(network, 32, 9, 1)
		network = style_conv_block(network, 64, 9, 2)
		network = style_conv_block(network, 128, 9, 2)
		for _ in range(5):
			network = residual_block(network)
		network = nn_upsample(network)
		network = nn_upsample(network)
		network = style_conv_block(network, 3, 9, 1, sigmoid)
		return network

	def feature_loss(self, out_layer, target_layer):
		return T.mean(T.sqr(out_layer - target_layer))

	def batched_gram(self, fmap):
		fmap=fmap.flatten(ndim=3)
		return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape)

	def style_loss(self, out_layer, target_layer):
		return T.mean(T.sum(T.sqr(batched_gram(out_layer) - batched_gram(target_layer)), axis=(1,2)))