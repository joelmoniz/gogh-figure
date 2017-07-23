#!/usr/bin/env python
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
from lasagne.nonlinearities import sigmoid, tanh
from lasagne.layers import batch_norm, BatchNormLayer

from utils import *
from layers import *

try:
	REPO_DIR = __file__[:-1*__file__[::-1].index('/')]
except Exception:
	REPO_DIR = './'

class Network(object):

	LOSS_NET_VERSION = 0.1

	MODEL_PATH = REPO_DIR + 'data/model/'
	LOSS_NET_MODEL_FILE_NAME = "vgg16_loss_net.npz"
	LOSS_NET_MODEL_SIZE = 58863490
	LOSS_NET_DOWNLOAD_LINK = "TODO" + str(LOSS_NET_VERSION) + "TODO" + LOSS_NET_MODEL_FILE_NAME
	LOSS_NET_MODEL_FILE_PATH = MODEL_PATH + LOSS_NET_MODEL_FILE_NAME

	def __init__(self, input_var=None, num_styles=None, shape=(None, 3, 256, 256), net_type=1, **kwargs):
		"""
		net_type: 0 (fast neural style- fns) or 1 (conditional instance norm- cin)
		"""
		assert net_type in [0, 1]
		self.net_type = net_type
		self.network = {}

		if len(shape) == 2:
			shape=(None, 3, shape[0], shape[1])
		elif len(shape) == 3:
			shape=(None, shape[0], shape[1], shape[2])
		self.shape = shape

		self.num_styles = num_styles

		self.network['loss_net'] = {}
		self.setup_loss_net()
		self.load_loss_net_weights()

		self.network['transform_net'] = {}
		self.setup_transform_net(input_var)


	def setup_loss_net(self):
		"""
		Create a network of convolution layers based on the VGG16 architecture from the paper:
		"Very Deep Convolutional Networks for Large-Scale Image Recognition"

		Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
		License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

		Based on code in the Lasagne Recipes repository: https://github.com/Lasagne/Recipes
		"""
		loss_net = self.network['loss_net']
		loss_net['input'] = InputLayer(shape=self.shape)
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
		transform_net = InputLayer(shape=self.shape, input_var=input_var)
		transform_net = style_conv_block(transform_net, self.num_styles, 32, 9, 1)
		transform_net = style_conv_block(transform_net, self.num_styles, 64, 3, 2)
		transform_net = style_conv_block(transform_net, self.num_styles, 128, 3, 2)
		for _ in range(5):
			transform_net = residual_block(transform_net, self.num_styles)
		transform_net = nn_upsample(transform_net, self.num_styles)
		transform_net = nn_upsample(transform_net, self.num_styles)

		if self.net_type == 0:
			transform_net = style_conv_block(transform_net, self.num_styles, 3, 9, 1, tanh)
			transform_net = ExpressionLayer(transform_net, lambda X: 150.*X, output_shape=None)
		elif self.net_type == 1:
			transform_net = style_conv_block(transform_net, self.num_styles, 3, 9, 1, sigmoid)

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
		if self.net_type == 0:
			return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape[-2:])
		elif self.net_type == 1:
			return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape[-1])

	def style_loss(self, out_layer, target_style_layer):
		# Each input is a 4D tensor: (batch, feature map, height, width)
		# TODO: Make the first dim broadcastable instead of tiling
		return T.mean(T.sqr(self.batched_gram(out_layer) - T.tile(self.batched_gram(target_style_layer), (T.shape(out_layer)[0], 1, 1))))

	def style_loss_pg(self, out_layer, target_style_gram):
		# Each input is a 4D tensor: (batch, feature map, height, width)
		# TODO: Make the first dim broadcastable instead of tiling
		return T.mean(T.sqr(self.batched_gram(out_layer) - T.tile(target_style_gram, (T.shape(out_layer)[0], 1, 1))))

	def total_variation_loss(self, x):
		# https://github.com/alexjc/neural-enhance/blob/master/enhance.py#L408-L409
		return T.sum(((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25)

class CocoData(object):

	PREPARE_COCO_SCRIPT_NAME = ''

	def __init__(self, h5_location=None, train_batchsize=16, valid_batchsize=16):
		if h5_location == None:
			h5_location = REPO_DIR + "data/content/ms-coco-256.h5"
		if not path_exists(h5_location):
			print("Please download the COCO dataset, run the " +
				self.PREPARE_COCO_SCRIPT_NAME + " and ensure that the hdf5 file is at " +
				h5_location)

		self.dataset = h5py.File(h5_location, "r")
		self.train_batchsize = train_batchsize
		self.valid_batchsize = valid_batchsize

		self.vgg_mean = [103.939, 116.779, 123.68]

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

			yield self.preprocess_vgg(inputs[excerpt])

	def preprocess_range(self, original):
		return (original/np.asarray(255., theano.config.floatX))

	def depreprocess_range(self, processed):
		return processed*np.asarray(255, dtype='uint8')

	def preprocess_vgg(self, original):
		return original[:,::-1,:,:]-np.asarray(np.reshape(self.vgg_mean,(1,3,1,1)), dtype=theano.config.floatX)
		scale = 255. if is_scaled_down else 1.
		scale = np.asarray(scale, dtype=theano.config.floatX)
		if len(original.shape) == 4:
			return np.asarray(scale*original[:,::-1,:,:]-np.reshape(self.vgg_mean,(1,3,1,1)), dtype=theano.config.floatX)
		elif len(original.shape) == 3:
			return np.asarray(scale*original[::-1,:,:]-np.reshape(self.vgg_mean,(3,1,1)), dtype=theano.config.floatX)

	def deprocess_vgg(self, processed):
		return (processed+np.reshape(self.vgg_mean,(1,3,1,1)))[:,::-1,:,:]

	def range_to_vgg(self, range_processed):
		return self.preprocess_vgg(self.depreprocess_range(range_processed))

	def vgg_to_range(self, vgg_processed):
		return self.preprocess_range(self.deprocess_vgg(vgg_processed))

	def get_train_batch(self):
		return self.iterate_minibatches(self.dataset['train2014']['images'], self.train_batchsize, True)

	def get_valid_batch(self):
		return self.iterate_minibatches(self.dataset['val2014']['images'], self.valid_batchsize, False)

	def get_first_valid_batch(self):
		return self.preprocess_vgg(self.dataset['val2014']['images'][:self.valid_batchsize])
