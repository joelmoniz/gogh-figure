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
from lasagne.nonlinearities import softmax, rectify, elu, sigmoid, tanh, softplus
from lasagne.layers import batch_norm, BatchNormLayer

from utils import *
from layers import *

try:
	REPO_DIR = __file__[:-1*__file__[::-1].index('/')]
except Exception, e:
	REPO_DIR = './'

class Network(object):

	LOSS_NET_VERSION = 0.1

	MODEL_PATH = REPO_DIR + 'data/model/'
	LOSS_NET_MODEL_FILE_NAME = "vgg16_loss_net.npz"
	LOSS_NET_MODEL_SIZE = 58863490
	LOSS_NET_DOWNLOAD_LINK = "TODO" + str(LOSS_NET_VERSION) + "TODO" + LOSS_NET_MODEL_FILE_NAME
	LOSS_NET_MODEL_FILE_PATH = MODEL_PATH + LOSS_NET_MODEL_FILE_NAME

	def __init__(self, input_var=None, shape=(None, 3, 256, 256), net_type=1, **kwargs):
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
		transform_net = style_conv_block(transform_net, 32, 9, 1)
		transform_net = style_conv_block(transform_net, 64, 9, 2)
		transform_net = style_conv_block(transform_net, 128, 9, 2)
		for _ in range(5):
			transform_net = residual_block(transform_net)
		transform_net = nn_upsample(transform_net)
		transform_net = nn_upsample(transform_net)

		if self.net_type == 0:
			transform_net = style_conv_block(transform_net, 3, 9, 1, tanh)
			transform_net = ExpressionLayer(transform_net, lambda X: 150.*X, output_shape=None)
		elif self.net_type == 1:
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
		if self.net_type == 0:
			return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape[-2:])
		elif self.net_type == 1:
			return T.batched_dot(fmap, fmap.dimshuffle(0,2,1))/T.prod(fmap.shape[-1])

	def style_loss(self, out_layer, target_style_layer):
		# Each input is a 4D tensor: (batch, feature map, height, width)
		return T.mean(T.sqr(self.batched_gram(out_layer) - T.tile(self.batched_gram(target_style_layer), (T.shape(out_layer)[0], 1, 1))))

	def total_variation_loss(self, x):
		# https://github.com/alexjc/neural-enhance/blob/master/enhance.py#L408-L409
		return T.mean(((x[:,:,:-1,:-1] - x[:,:,1:,:-1])**2 + (x[:,:,:-1,:-1] - x[:,:,:-1,1:])**2)**1.25)

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


def train():
	# TODO: These should be user accpeted:
	DEBUG = True
	VALIDATE = False
	STYLE_LOSS_LAYERS = {'conv1_2': 4e-4,'conv2_2': 4e-4,'conv3_3': 4e-4,'conv4_3': 4e-4}
	TOTAL_VARIATION_LOSS_WEIGHT = 0.
	CONTENT_LOSS_LAYER = 'conv3_3'
	NUM_EPOCHS = 2 # 40k steps is around 8 epochs
	STYLE_IMAGE_LOCATION = REPO_DIR + 'data/images/styles/candy.jpg'
	FOLDER_SUFFIX = 'jc_s5_ve-6_i_candy'

	create_dir_if_not_exists(REPO_DIR + 'data/model/trained_' + FOLDER_SUFFIX)
	create_dir_if_not_exists(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX)

	image_var = T.tensor4('inputs')
	pastiche_content_var = T.tensor4('pastiche_content')
	style_var = T.TensorType(theano.config.floatX,(False,)*5) ('style')
	pastiche_style_var = T.TensorType(theano.config.floatX,(False,)*5) ('pastiche_style')
	chosen_style_var=T.ivector('chosen_style')

	print('Loading Data...')
	data = CocoData(train_batchsize=4)

	print('Loading Networks...')
	net = Network(data.vgg_to_range(image_var))

	print('Compiling Functions...')
	# initialize transformer network function
	transform_pastiche_out = lasagne.layers.get_output(net.network['transform_net'], style=chosen_style_var)
	pastiche_transform_fn = theano.function([image_var, chosen_style_var], transform_pastiche_out)

	# initialize loss network related functions
	style_loss_layer_keys = STYLE_LOSS_LAYERS.keys()
	style_loss_layer_weights = [STYLE_LOSS_LAYERS[w]/(1.0*len(style_loss_layer_keys)) for w in style_loss_layer_keys]

	if CONTENT_LOSS_LAYER in style_loss_layer_keys:
		vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys], data.range_to_vgg(transform_pastiche_out))
		vgg_pastiche_style_out = vgg_all_out
		vgg_pastiche_content_out = vgg_all_out[style_loss_layer_keys.index(CONTENT_LOSS_LAYER)]
	else:
		vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys+[CONTENT_LOSS_LAYER]], data.range_to_vgg(transform_pastiche_out))
		vgg_pastiche_style_out = vgg_all_out[:-1]
		vgg_pastiche_content_out = vgg_all_out[-1]
	vgg_style_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys], image_var)
	vgg_content_out = lasagne.layers.get_output(net.network['loss_net'][CONTENT_LOSS_LAYER], image_var)

	style_image_vgg_fn = theano.function([image_var], vgg_style_out)
	content_image_vgg_fn = theano.function([image_var], vgg_content_out)
	# pastiche_vgg_fn = theano.function([image_var], vgg_all_out)

	# Get the VGG16 Loss Network's representaion of the style image
	# style_im = np.expand_dims(data.preprocess_vgg(get_image(STYLE_IMAGE_LOCATION, (256, 256)), True), 0)
	style_im = data.range_to_vgg(get_images(STYLE_IMAGE_LOCATION, (256, 256), maintain_aspect=True))
	print(style_im.shape)
	vgg_style_values = style_image_vgg_fn(style_im)

	# Initialize loss functions
	content_loss_value = net.feature_loss(vgg_pastiche_content_out, vgg_content_out)
	style_loss_value = 0.
	for pso, vsv, sllw in zip(vgg_pastiche_style_out, vgg_style_values, style_loss_layer_weights):
		style_loss_value += net.style_loss(pso, theano.shared(vsv)[[chosen_style_var[0]]])*sllw
	total_variation_loss_value = TOTAL_VARIATION_LOSS_WEIGHT * net.total_variation_loss(transform_pastiche_out)
	loss = content_loss_value + style_loss_value + total_variation_loss_value

	params = lasagne.layers.get_all_params(net.network['transform_net'], trainable=True)
	updates = lasagne.updates.adam(loss, params)
	train_fn = theano.function([image_var, chosen_style_var], [loss, content_loss_value, style_loss_value, total_variation_loss_value], updates=updates)

	# TODO: If using conditional instance norm, make this deterministic
	# valid_fn = theano.function([image_var], loss)

	if DEBUG:
		content_ims = data.get_first_valid_batch()
		save_params(REPO_DIR + 'data/model/trained_' + FOLDER_SUFFIX + '/e0.npz', net.network['transform_net'])
		save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, data.deprocess_vgg(content_ims), 'orig_im')
		for i in range(0, style_im.shape[0]):
			save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, pastiche_transform_fn(content_ims, [i]*data.valid_batchsize), 'e0_s'+str(i)+'_im')

	print('Commencing Training...')
	# For each epoch
	for epoch in range(NUM_EPOCHS):

		train_err = 0
		valid_err = 0
		train_batch_num = 0
		valid_batch_num = 0
		total_batch_err = content_loss_err= style_loss_err= total_variation_loss_err = 0
		start_time = time.time()

		for content_ims in data.get_train_batch():
			batch_style_num = np.random.randint(0, style_im.shape[0])
			batch_train_err, minib_content_loss_err, minib_style_loss_err, minib_total_variation_loss_err = train_fn(content_ims, [batch_style_num]*data.train_batchsize)
			train_err += batch_train_err
			total_batch_err += batch_train_err
			content_loss_err += minib_content_loss_err
			style_loss_err += minib_style_loss_err
			total_variation_loss_err += minib_total_variation_loss_err
			train_batch_num += 1

			if DEBUG and train_batch_num%400 == 0:
				print("Batch " + str(train_batch_num) + ":\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(train_err / train_batch_num, total_batch_err/400, content_loss_err/400, style_loss_err/400, total_variation_loss_err/400))
				total_batch_err= content_loss_err= style_loss_err= total_variation_loss_err = 0
				if train_batch_num%min(400*style_im.shape[0], 25000) == 0:
					for i in range(0, style_im.shape[0]):
						save_im(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX + '/e' + str(epoch + 1) + 'b' + str(train_batch_num) + 's' + str(i) + '.jpg', pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize))

		if DEBUG:
			for i in range(0, style_im.shape[0]):
				save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize), 'e' + str(epoch + 1) + '_s' + str(i) + '_im')

		save_params(REPO_DIR + 'data/model/trained_' + FOLDER_SUFFIX + '/e' + str(epoch + 1) + '.npz', net.network['transform_net'])
		# if VALIDATE:
		# 	for content_ims in data.get_valid_batch():
		# 		valid_err += valid_fn(content_ims)
		# 		valid_batch_num += 1

		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, NUM_EPOCHS, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batch_num))
		if VALIDATE:
			print("  valid loss:\t\t{:.6f}".format(valid_err / valid_batch_num))

def stylize():
	# TODO: These should be user accpeted:
	INPUT_LOCATION = 'testims/'
	OUTPUT_LOCATION = 'testims_out128/'
	MODEL_FILE = 'e2.npz'
	shape = (256, 256)

	image_var = T.tensor4('inputs')

	print('Loading Networks...')
	net = Network(image_var, shape)
	load_params(net.network['transform_net'], MODEL_FILE)

	print('Loading Images...')
	data = CocoData(train_batchsize=4)
	ims = data.preprocess_vgg(get_images(INPUT_LOCATION, shape), True)

	print('Compiling Functions...')
	# initialize transformer network function
	transform_pastiche_out = lasagne.layers.get_output(net.network['transform_net'])
	pastiche_transform_fn = theano.function([image_var], transform_pastiche_out)

	print('Transforming images...')
	out_ims = pastiche_transform_fn(ims)

	print('Saving images...')
	save_ims(OUTPUT_LOCATION, data.deprocess_vgg(out_ims))

	print('Done.')

if __name__ == '__main__':
	train()