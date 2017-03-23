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

from utils import *
from model import *

import argparse
import yaml

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--debug", type=int, choices=[0,1], default=1,
						help="display loss information and save intermediate images when training")
	parser.add_argument("-v", "--validate", type=int, choices=[0,1], default=0,
						help="calculate validation loss after each epoch (warning: increases train time)")
	parser.add_argument("-n", "--network", type=int, choices=[0,1], default=1,
						help="transform network architecture: 0- fast neural style; 1- conditional instance norm")
	# That's nice: http://stackoverflow.com/a/20493276
	parser.add_argument('-s', '--styleloss', type=yaml.load, default="{'conv1_2': 4e-4,'conv2_2': 4e-4,'conv3_3': 4e-4,'conv4_3': 4e-4}",
						help="a dict with (layer, weight) mappings")
	parser.add_argument('-t', '--varloss', type=float, default=0.,
						help="the weight of the total variational loss")
	parser.add_argument('-c', '--contentloss', type=str, default='conv3_3',
						help="the content loss layer; weight fixed at 1.0")
	parser.add_argument("-e", "--epochs", type=int, default=2,
						help="the number of epochs to train the system")
	parser.add_argument("-b", "--batchsize", type=int, default=4,
						help="the batchsize to be used during training")
	parser.add_argument('-i', '--styleloc', type=str, default=REPO_DIR + 'data/styles/candy.jpg',
						help="the file to be used as the style image, or the folder containing all the style images")
	parser.add_argument('-a', '--suffix', type=str, default='jc_s5_ve-6_i_candy',
						help="the suffix to be added to the folders used to store debug images and trained model params")

	args = parser.parse_args()

	# Needed because YAML fails to parse 4e-4 : http://stackoverflow.com/a/30462009
	for k in args.styleloss:
		args.styleloss[k]=float(args.styleloss[k])

	return args

def train(args):
	DEBUG = args.debug
	VALIDATE = args.validate
	NET_TYPE = args.network
	STYLE_LOSS_LAYERS = args.styleloss
	TOTAL_VARIATION_LOSS_WEIGHT = args.varloss
	CONTENT_LOSS_LAYER = args.contentloss
	NUM_EPOCHS = args.epochs
	STYLE_IMAGE_LOCATION = args.styleloc
	FOLDER_SUFFIX = args.suffix

	create_dir_if_not_exists(REPO_DIR + 'data/model/trained_' + FOLDER_SUFFIX)
	create_dir_if_not_exists(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX)

	image_var = T.tensor4('inputs')
	pastiche_content_var = T.tensor4('pastiche_content')
	style_var = T.TensorType(theano.config.floatX,(False,)*5) ('style')
	pastiche_style_var = T.TensorType(theano.config.floatX,(False,)*5) ('pastiche_style')
	chosen_style_var=T.ivector('chosen_style')

	print('Loading Data...')
	data = CocoData(train_batchsize=4)
	style_im = get_images(STYLE_IMAGE_LOCATION, (256, 256), maintain_aspect=True)

	print('Loading Networks...')
	if NET_TYPE == 1:
		net = Network(data.vgg_to_range(image_var), len(style_im))
	elif NET_TYPE == 0:
		net = Network(image_var, len(style_im), net_type=0)

	print('Compiling Functions...')
	# initialize transformer network function
	transform_pastiche_out = lasagne.layers.get_output(net.network['transform_net'], style=chosen_style_var)
	pastiche_transform_fn = theano.function([image_var, chosen_style_var], transform_pastiche_out)

	# initialize loss network related functions
	style_loss_layer_keys = STYLE_LOSS_LAYERS.keys()
	style_loss_layer_weights = [STYLE_LOSS_LAYERS[w]/(1.0*len(style_loss_layer_keys)) for w in style_loss_layer_keys]

	if CONTENT_LOSS_LAYER in style_loss_layer_keys:
		if NET_TYPE == 1:
			vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys], data.range_to_vgg(transform_pastiche_out))
		elif NET_TYPE == 0:
			vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys], transform_pastiche_out)
		vgg_pastiche_style_out = vgg_all_out
		vgg_pastiche_content_out = vgg_all_out[style_loss_layer_keys.index(CONTENT_LOSS_LAYER)]
	else:
		if NET_TYPE == 1:
			vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys+[CONTENT_LOSS_LAYER]], data.range_to_vgg(transform_pastiche_out))
		elif NET_TYPE == 0:
			vgg_all_out = lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys+[CONTENT_LOSS_LAYER]], transform_pastiche_out)
		vgg_pastiche_style_out = vgg_all_out[:-1]
		vgg_pastiche_content_out = vgg_all_out[-1]
	vgg_style_gram_out = [net.batched_gram(vs) for vs in lasagne.layers.get_output([net.network['loss_net'][x] for x in style_loss_layer_keys], image_var)]
	vgg_content_out = lasagne.layers.get_output(net.network['loss_net'][CONTENT_LOSS_LAYER], image_var)

	style_image_vgg_fn = theano.function([image_var], vgg_style_gram_out)
	content_image_vgg_fn = theano.function([image_var], vgg_content_out)
	# pastiche_vgg_fn = theano.function([image_var], vgg_all_out)

	# Get the VGG16 Loss Network's representaion of the style image
	# style_im = np.expand_dims(data.preprocess_vgg(get_image(STYLE_IMAGE_LOCATION, (256, 256)), True), 0)
	style_ims_gram = [style_image_vgg_fn(data.range_to_vgg(np.expand_dims(im, axis=0))) for im in style_im]
	# pdb.set_trace()
	style_ims_gram = list(map(list, zip(*style_ims_gram))) # gotta love python; gotta love SO even more: http://stackoverflow.com/a/6473724/2427542
	style_ims_gram = [np.asarray(s) for s in style_ims_gram]
	# pdb.set_trace()

	# Initialize loss functions
	content_loss_value = net.feature_loss(vgg_pastiche_content_out, vgg_content_out)
	style_loss_value = 0.
	for pso, vsv, sllw in zip(vgg_pastiche_style_out, style_ims_gram, style_loss_layer_weights):
		style_loss_value += net.style_loss_pg(pso, theano.shared(vsv)[chosen_style_var[0]])*sllw
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
		for i in range(0, len(style_im)):
			if NET_TYPE == 1:
				save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, pastiche_transform_fn(content_ims, [i]*data.valid_batchsize), 'e0_s'+str(i)+'_im')
			elif NET_TYPE == 0:
				save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, data.deprocess_vgg(pastiche_transform_fn(content_ims, [i]*data.valid_batchsize)), 'e0_s'+str(i)+'_im')

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
			batch_style_num = np.random.randint(0, len(style_im))
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
				if train_batch_num%min(400*len(style_im), 5000) == 0:
					for i in range(0, len(style_im)):
						if NET_TYPE == 1:
							save_im(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX + '/e' + str(epoch + 1) + 'b' + str(train_batch_num) + 's' + str(i) + '.jpg', pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize))
						elif NET_TYPE == 0:
							save_im(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX + '/e' + str(epoch + 1) + 'b' + str(train_batch_num) + 's' + str(i) + '.jpg', data.deprocess_vgg(pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize)))

		if DEBUG:
			for i in range(0, len(style_im)):
				if NET_TYPE == 1:
					save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize), 'e' + str(epoch + 1) + '_s' + str(i) + '_im')
				elif NET_TYPE == 0:
					save_ims(REPO_DIR + 'data/debug/im_' + FOLDER_SUFFIX, data.deprocess_vgg(pastiche_transform_fn(data.get_first_valid_batch(), [i]*data.valid_batchsize)), 'e' + str(epoch + 1) + '_s' + str(i) + '_im')

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

if __name__ == '__main__':
	args = parse_args()
	train(args)
