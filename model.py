from __future__ import print_function

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
import h5py

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
from layers import *

class Network(object):

	LOSS_NET_VERSION = 0.1

	MODEL_PATH = __file__[:-1*__file__[::-1].index('/')] + 'data/model/'
	LOSS_NET_MODEL_FILE_NAME = "vgg16_loss_net.npz"
	LOSS_NET_MODEL_SIZE = 58863490
	LOSS_NET_DOWNLOAD_LINK = "TODO" + str(LOSS_NET_VERSION) + "TODO" + LOSS_NET_MODEL_FILE_NAME
	LOSS_NET_MODEL_FILE_PATH = MODEL_PATH + LOSS_NET_MODEL_FILE_NAME

	def __init__(self, input_var, **kwargs):
		self.network = {}

		self.network['loss_net'] = {}
		self.setup_loss_net(input_var)
		self.load_loss_net_weights()

		self.network['transform_net'] = {}
		self.setup_transform_net(input_var)

	def setup_loss_net(self, input_var):
		"""
		Create a network of convolution layers based on the VGG16 architecture from the paper:
		"Very Deep Convolutional Networks for Large-Scale Image Recognition"

		Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
		License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

		Based on code in the Lasagne Recipes repository: https://github.com/Lasagne/Recipes
		"""
		loss_net = self.network['loss_net']
		loss_net['input'] = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
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
		download_if_not_exists(self.LOSS_NET_MODEL_FILE_PATH, self.LOSS_NET_DOWNLOAD_LINK, \
			"Downloading the Loss Network's weights", self.LOSS_NET_MODEL_SIZE)
		load_params(self.network['loss_net']['conv5_3'], self.LOSS_NET_MODEL_FILE_PATH)

	def setup_transform_net(self, input_var=None):
		transform_net = InputLayer(shape=(None, 3, 256, 256), input_var=input_var)
		transform_net = style_conv_block(transform_net, 32, 9, 1)
		transform_net = style_conv_block(transform_net, 64, 9, 2)
		transform_net = style_conv_block(transform_net, 128, 9, 2)
		for _ in range(5):
			transform_net = residual_block(transform_net)
		transform_net = nn_upsample(transform_net)
		transform_net = nn_upsample(transform_net)
		transform_net = style_conv_block(transform_net, 3, 9, 1, sigmoid)

		self.network['transform_net'] = transform_net

	def feature_loss(self, out_layer, target_layer):
		return T.mean(T.sqr(out_layer - target_layer))

	def batched_gram5d(self, fmap):
		# (layer, batch, featuremaps, height*width)
		fmap=fmap.flatten(ndim=4)

		# (layer*batch, featuremaps, height*width)
		fmap2=fmap.reshape((-1, fmap.shape[-2], fmap.shape[-1]))

		# The T.prod term can't be taken outside as a T.mean in style_loss(), since the width and height of the image might vary
		return T.batched_dot(fmap2, fmap2.dimshuffle(0,2,1)).reshape(fmap.shape)/T.prod(fmap.shape[-2:])

	def style_loss5d(self, out_layer, target_style_layer):
		# Each input is a 5D tensor: (style loss layer, batch, feature map, height, width)
		return T.mean(T.sum(T.sqr(self.batched_gram(out_layer) - T.tile(self.batched_gram(target_style_layer), (1, T.shape(out_layer)[0], 1, 1))), axis=(2,3)), axis=1)

	def batched_gram(self, fmap):
		# (batch, featuremaps, height*width)
		fmap=fmap.flatten(ndim=3)

		# The T.prod term can't be taken outside as a T.mean in style_loss(), since the width and height of the image might vary
		return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape[-2:])

	def style_loss(self, out_layer, target_style_layer):
		# Each input is a 4D tensor: (batch, feature map, height, width)
		return T.mean(T.sum(T.sqr(self.batched_gram(out_layer) - T.tile(self.batched_gram(target_style_layer), (T.shape(out_layer)[0], 1, 1))), axis=(1,2)), axis=0)

class CocoData(object):

	PREPARE_COCO_SCRIPT_NAME = ''

	def __init__(self, h5_location=None, train_batchsize=16, valid_batchsize=16):
		if h5_location == None:
			h5_location = REPO_DIR + "data/images/content/ms-coco-256.h5"
		if not path_exists(h5_location):
			print("Please download the COCO dataset, run the " +
				self.PREPARE_COCO_SCRIPT_NAME + " and ensure that the hdf5 file is at " +
				h5_location)

		self.dataset = h5py.File(h5_location, "r")
		self.train_batchsize = train_batchsize
		self.valid_batchsize = valid_batchsize

	def iterate_minibatches(self, inputs, batchsize, shuffle=False):
		if shuffle:
			indices = np.arange(len(inputs))
			np.random.shuffle(indices)
		for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
			if shuffle:
				excerpt = list(indices[start_idx:start_idx + batchsize])
				excerpt.sort()
			else:
				excerpt = slice(start_idx, start_idx + batchsize)

			yield inputs[excerpt]

	def get_train_batch(self):
		return self.iterate_minibatches(self.dataset['train2014']['images'], self.train_batchsize, True)

	def get_valid_batch(self):
		return self.iterate_minibatches(self.dataset['val2014']['images'], self.valid_batchsize, False)
