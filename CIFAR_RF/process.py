import numpy as np
import matplotlib.pyplot as plt
import cPickle

DATA_PATH = "data/"
FILE_PREFIX = "data_batch"

def unpack_data(data_path, file_prefix):
	data = np.array()
	for fname in os.listdirs(data_path):
		if file_prefix not in fname:
			continue
		with tmp as open(data_path + fname, 'rb'):
			np.append( cPickle.load(tmp) )

print unpack_data(DATA_PATH, FILE_PREFIX).shape