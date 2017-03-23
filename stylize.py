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
import ast

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--inputloc', type=str, default='gogh-fig-pics/test_input/',
						help="the path to the image, or to a folder containing the images, to be stylized")
	parser.add_argument('-o', '--outputloc', type=str, default='gogh-fig-pics/test_output/',
						help="the location where the stylized image/images is/are to be stored")
	parser.add_argument('-m', '--modelloc', type=str, default='data/model/trained_cin_se-4_i2c_c33_ea_misc32/e2.npz',
						help="the location of the trained model file")
	parser.add_argument("-n", "--numstyles", type=int, default=32,
						help="the number of styles in the trained model file, i.e., the number of style images used to train the loaded model")
	parser.add_argument("-b", "--batchsize", type=int, default=4,
						help="the batchsize to be used during stylization")
	parser.add_argument("-d", "--dim", type=str, default=None,
						help="the (rough) output dimension; None (default) leaves the images the same size as the input; providing a tuple makes the pastiche roughly this size; (975, 1300) seems to work well")

	args = parser.parse_args()

	if args.dim:
		args.dim = ast.literal_eval(args.dim)

	return args

def stylize(args):
	INPUT_LOCATION = args.inputloc
	OUTPUT_LOCATION = args.outputloc
	MODEL_FILE = args.modelloc
	NUM_STYLES = args.numstyles
	MAX_BATCH_SIZE = args.batchsize
	DIMENSION = args.dim

	image_var = T.tensor4('inputs')
	chosen_style_var = T.ivector('chosen_style')

	print('Loading Images...')
	ims = get_images(INPUT_LOCATION, dim=DIMENSION, center=False, correct_vertical=True)

	print('Loading Networks...')
	net = Network(image_var, NUM_STYLES, (None,3,None,None))
	load_params(net.network['transform_net'], MODEL_FILE)

	print('Compiling Functions...')
	# initialize transformer network function
	transform_pastiche_out = lasagne.layers.get_output(net.network['transform_net'], style=chosen_style_var)
	pastiche_transform_fn = theano.function([image_var, chosen_style_var], transform_pastiche_out)
	print('Transforming images...')

	# TODO: There might be a more efficient way to do this if there's only one style image,
	# but it would require all the content images to be of the same size to get batched together
	for num, im in enumerate(ims):
		out_ims = []
		start_time = time.time()
		imb = np.expand_dims(im, axis=0)
		imb=np.tile(imb, (MAX_BATCH_SIZE,1,1,1))
		for i in range(NUM_STYLES/MAX_BATCH_SIZE):
			out_ims += list(pastiche_transform_fn(imb, range(MAX_BATCH_SIZE*i, MAX_BATCH_SIZE*(i+1))))
		out_ims += list(pastiche_transform_fn(imb, range(MAX_BATCH_SIZE*(NUM_STYLES/MAX_BATCH_SIZE), NUM_STYLES)))
		print("  Done with image {}. Time: {:.3f}s".format(num, time.time() - start_time))
		save_ims(OUTPUT_LOCATION, out_ims, 'im' + str(num) + '_')
		print('  Saved.')

	print('Done.')


if __name__ == '__main__':
	args = parse_args()
	stylize(args)
