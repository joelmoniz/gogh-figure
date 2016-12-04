import os
from tqdm import tqdm
import requests
import numpy as np
import lasagne

def create_dir_if_not_exists(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

def download_if_not_exists(file_path, download_link, message=None, total_size=None):
	if os.path.exists(file_path):
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
	with np.load(model_file) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)