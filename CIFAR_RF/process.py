import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import cPickle
import os

DATA_PATH = "data/"
FILE_PREFIX = "data_batch"
META_FILE = "batches.meta"
label_names = []

def show_image(data):
	tmp = data.reshape(3, 32, 32).transpose(1, 2, 0)
	plt.imshow(tmp)
	plt.show()

def get_label_names(fname):
	with open(fname, 'rb') as f:
		ln = cPickle.load(f)
		return ln['label_names']

def unpack_data(data_path, file_prefix):	
	global label_names
	count, data, labels = 0, np.empty(shape=(50000, 3072)), np.empty(shape=(50000))
	for fname in os.listdir(data_path):
		if file_prefix not in fname:
			continue
		tmp = open(data_path + fname, 'rb')
		d = cPickle.load(tmp)
		data[count*10000:(count+1)*10000, :] = d['data']
		labels[count*10000:(count+1)*10000] = np.array(d['labels'])
		# show_image(data[count*10000,:])
		# print data[count*10000, :] == d['data'][0,:]
		count += 1
		print "Processing batch",count
		# print d['data'].shape
		# for k in d.keys():
		# 	print k,type(d[k])
		# print label_names[ d['labels'][1] ]
		# show_image(d['data'][0,:])
		tmp.close()
	print "Finished Processing"
	return data, labels

label_names = get_label_names(DATA_PATH + META_FILE)
train, labels = unpack_data(DATA_PATH,FILE_PREFIX)
print train.shape, labels.shape
print "Initializing random forest"
rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
print "Fitting data"
rf.fit(train, labels)
print "Training complete"
# show_image(train[0,:])