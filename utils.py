import os
from tqdm import tqdm
import requests
import numpy as np
import lasagne
from cv2 import resize
import cv2
from scipy.misc import imsave
from shutil import copy

def path_exists(path):
	return os.path.exists(path)

def create_dir_if_not_exists(directory):
	if not path_exists(directory):
		os.makedirs(directory)

def copy_file_if_not_exists(source_file, dest_dir):
	if path_exists(dest_dir + '/' + source_file.split('/')[-1]):
		return
	create_dir_if_not_exists(dest_dir)
	copy(source_file, dest_dir)

def download_if_not_exists(file_path, download_link, message=None, total_size=None):
	if path_exists(file_path):
		return

	if message != None:
		print(message)

	create_dir_if_not_exists('/'.join(file_path.split('/')[:-1]))
	download(file_path, download_link, total_size)

def download(file_path, download_link, total_size):
	"""
	Based on code in this answer: http://stackoverflow.com/a/10744565/2427542
	"""
	response = requests.get(download_link, stream=True)
	with open(file_path, "wb") as handle:
		for data in tqdm(response.iter_content(), total=total_size):
			handle.write(data)

def load_params(network, model_file):
	assert path_exists(model_file)
	with np.load(model_file) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

def save_params(file_name, network):
	np.savez(file_name, *lasagne.layers.get_all_param_values(network))

def get_image(path, dim=None, grey=False, maintain_aspect=True, center=True):
	"""
	Given an image path, return a 3D numpy array with the image. Maintains aspect ratio and center crops the image to match dim.
	:type path: str
	:param path: The location of the image
	:type grey: boolean
	:param grey: Whether the image should be returned in greyscale
	:type dim: tuple
	:param dim: The (height, width)
	"""
	assert path_exists(path)

	if grey:
		im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		im = np.expand_dims(im, axis=-1)
	else:
		# dimensions are (height, width, channel)
		im = cv2.imread(path, cv2.IMREAD_COLOR)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

	im = im.astype('float32')
	im = im / 255.

	if dim != None:
		if not maintain_aspect:
			im = resize(im, dim)
		else:
			im = resize_maintain_aspect(im, dim)
			if center:
				im = center_crop(im, dim)

	im = im.transpose(2, 0, 1)
	return im

def get_image_as_batch(path, **kwargs):
	return np.expand_dims(get_image(path, **kwargs), axis=0)

def get_images(path, dim=(256, 256), grey=False, **kwargs):
	"""
	Given a folder, return a 4D numpy array with all images in the folder
	"""
	path += '/'
	ims_paths = [path+im_path for im_path in os.listdir(path) if os.path.isfile(path+im_path)]

	if grey:
		ims = np.zeros((len(ims_paths), 1, dim[0], dim[1]), dtype='float32')
	else:
		ims = np.zeros((len(ims_paths), 3, dim[0], dim[1]), dtype='float32')

	for i, im_path in enumerate(ims_paths):
		ims[i] = get_image(im_path, dim, grey, **kwargs)

	return ims

def resize_maintain_aspect(im, dim):
	"""
	Resize an image while maintaining its aspect ratio. Resizes the smaller side of the image
	to match the corresponding dimension length specified.
	"""
	# The reversal of get_aspect_maintained_dim()'s output is needed because OpenCV's
	# resize() method takes the new size in the form (x, y)
	return resize(im, get_aspect_maintained_dim(im.shape, dim)[::-1])

def get_aspect_maintained_dim(old_dim, new_dim):
	"""
	Given an image's dimension and the dimension to which it is to be resized, returns the dimension to which
	the image can be resized while maintaining its aspect ratio.
	"""
	if old_dim[1] < old_dim[0]:
		return (int(old_dim[0]*(new_dim[1]/(old_dim[1]*1.0))), new_dim[1])
	else:
		return (new_dim[0], int(old_dim[1]*(new_dim[0]/(old_dim[0]*1.0))))

def center_crop(im, dim):
	"""
	Center-crops a portion of dimensions `dim` from the image.
	"""
	r = max(0, (dim[0]-im.shape[0])//2)
	c = max(0, (dim[1]-im.shape[1])//2)
	return im[r:r+dim[0], c:c+dim[1], :]

def save_im(file_name, im):
	"""
	Saves an image in (channel, height, width) format.
	"""
	assert len(im.shape) == 4 or len(im.shape) == 3
	if len(im.shape) == 4:
		imsave(file_name, im[0].transpose(1, 2, 0))
	else:
		imsave(file_name, im.transpose(1, 2, 0))

def save_ims(folder_name, ims, prefix='im'):
	create_dir_if_not_exists(folder_name)
	for i, im in enumerate(ims):
		save_im(folder_name + '/' + prefix + str(i) + '.jpg', im)