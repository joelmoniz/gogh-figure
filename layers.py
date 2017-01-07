import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Layer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import batch_norm
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.init import Normal, Constant
from lasagne.nonlinearities import linear, rectify, sigmoid, identity

import numpy as np

class ReflectLayer(lasagne.layers.Layer):
	"""
	Layer for reflect padding. Based on code from https://gist.github.com/ajbrock/a3858c26282d9731191901b397b3ce9f
	"""

	def __init__(self, incoming, width, batch_ndim=2, **kwargs):
		super(ReflectLayer, self).__init__(incoming, **kwargs)
		self.width = width
		self.batch_ndim = batch_ndim

	def get_output_shape_for(self, input_shape):
		output_shape = list(input_shape)

		if isinstance(self.width, int):
			widths = [self.width] * (len(input_shape) - self.batch_ndim)
		else:
			widths = self.width

		for k, w in enumerate(widths):
			if output_shape[k + self.batch_ndim] is None:
				continue
			else:
				try:
					l, r = w
				except TypeError:
					l = r = w
				output_shape[k + self.batch_ndim] += l + r
		return tuple(output_shape)

	def get_output_for(self, input, **kwargs):
		return self.reflect_pad(input, self.width, self.batch_ndim)

	def reflect_pad(self, x, width, batch_ndim=1):
		"""
		Pad a tensor with a constant value.
		Parameters
		----------
		x : tensor
		width : int, iterable of int, or iterable of tuple
			Padding width. If an int, pads each axis symmetrically with the same
			amount in the beginning and end. If an iterable of int, defines the
			symmetric padding width separately for each axis. If an iterable of
			tuples of two ints, defines a seperate padding width for each beginning
			and end of each axis.
		batch_ndim : integer
			Dimensions before the value will not be padded.
		"""

		# Idea for how to make this happen: Flip the tensor horizontally to grab horizontal values, then vertically to grab vertical values
		# alternatively, just slice correctly
		input_shape = x.shape
		input_ndim = x.ndim

		output_shape = list(input_shape)
		indices = [slice(None) for _ in output_shape]

		if isinstance(width, int):
			widths = [width] * (input_ndim - batch_ndim)
		else:
			widths = width

		for k, w in enumerate(widths):
			try:
				l, r = w
			except TypeError:
				l = r = w
			output_shape[k + batch_ndim] += l + r
			indices[k + batch_ndim] = slice(l, l + input_shape[k + batch_ndim])

		# Create output array
		out = T.zeros(output_shape)

		# Vertical Reflections
		out=T.set_subtensor(out[:,:,:width,width:-width], x[:,:,width:0:-1,:])# out[:,:,:width,width:-width] = x[:,:,width:0:-1,:]
		out=T.set_subtensor(out[:,:,-width:,width:-width], x[:,:,-2:-(2+width):-1,:])#out[:,:,-width:,width:-width] = x[:,:,-2:-(2+width):-1,:]

		# Place X in out
		# out = T.set_subtensor(out[tuple(indices)], x) # or, alternative, out[width:-width,width:-width] = x
		out=T.set_subtensor(out[:,:,width:-width,width:-width],x)#out[:,:,width:-width,width:-width] = x

		#Horizontal reflections
		out=T.set_subtensor(out[:,:,:,:width],out[:,:,:,(2*width):width:-1])#out[:,:,:,:width] = out[:,:,:,(2*width):width:-1]
		out=T.set_subtensor(out[:,:,:,-width:],out[:,:,:,-(width+2):-(2*width+2):-1])#out[:,:,:,-width:] = out[:,:,:,-(width+2):-(2*width+2):-1]

		return out

class InstanceNormLayer(Layer):
	"""
	An implementation of Instance Normalization, based on lasagne's BatchNormLayer.
	Note that unlike the fns implementation, which uses 1e-5 as epsilon (the torch default),
	this implementation uses 1e-4 as epsilon (the lasagne default).
	"""
	def __init__(self, incoming, epsilon=1e-4,
				 beta=Constant(0), gamma=Constant(1), **kwargs):
		super(InstanceNormLayer, self).__init__(incoming, **kwargs)

		self.axes = (2, 3)
		self.epsilon = epsilon
		shape = (self.input_shape[1],)

		if beta is None:
			self.beta = None
		else:
			self.beta = self.add_param(beta, shape, 'beta',
									   trainable=True, regularizable=False)
		if gamma is None:
			self.gamma = None
		else:
			self.gamma = self.add_param(gamma, shape, 'gamma',
										trainable=True, regularizable=True)

	def get_output_for(self, input, **kwargs):

		mean = input.mean(self.axes)
		inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

		beta = 0 if self.beta is None else self.beta.dimshuffle(['x', 0, 'x', 'x'])
		gamma = 1 if self.gamma is None else self.gamma.dimshuffle(['x', 0, 'x', 'x'])
		mean = mean.dimshuffle([0, 1, 'x', 'x'])
		inv_std = inv_std.dimshuffle([0, 1, 'x', 'x'])

		# normalize
		normalized = (input - mean) * (gamma * inv_std) + beta
		return normalized

def instance_norm(layer, **kwargs):
	"""
	The equivalent of Lasagne's `batch_norm()` convenience method, but for instance normalization.
	Refer: http://lasagne.readthedocs.io/en/latest/modules/layers/normalization.html#lasagne.layers.batch_norm
	"""
	nonlinearity = getattr(layer, 'nonlinearity', None)
	if nonlinearity is not None:
		layer.nonlinearity = identity
	if hasattr(layer, 'b') and layer.b is not None:
		del layer.params[layer.b]
		layer.b = None
	bn_name = (kwargs.pop('name', None) or
			   (getattr(layer, 'name', None) and layer.name + '_bn'))
	layer = InstanceNormLayer(layer, name=bn_name, **kwargs)
	if nonlinearity is not None:
		nonlin_name = bn_name and bn_name + '_nonlin'
		layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
	return layer

# TODO: Add normalization
def style_conv_block(conv_in, num_filters, filter_size, stride, nonlinearity=rectify, normalization=batch_norm):
	sc_network = ReflectLayer(conv_in, filter_size//2)
	sc_network = normalization(ConvLayer(sc_network, num_filters, filter_size, stride, nonlinearity=nonlinearity, W=Normal()))
	return sc_network

def residual_block(resnet_in, num_filters=None, filter_size=3, stride=1):
	if num_filters == None:
		num_filters = resnet_in.output_shape[1]

	conv1 = style_conv_block(resnet_in, num_filters, filter_size, stride)
	conv2 = style_conv_block(conv1, num_filters, filter_size, stride, linear)
	res_block = ElemwiseSumLayer([conv2, resnet_in])

	return res_block

def nn_upsample(upsample_in, num_filters=None, filter_size=3, stride=1):
	if num_filters == None:
		num_filters = upsample_in.output_shape[1]

	nn_network = ExpressionLayer(upsample_in, lambda X: X.repeat(2, 2).repeat(2, 3), output_shape='auto')
	nn_network = style_conv_block(nn_network, num_filters, filter_size, stride)

	return nn_network
