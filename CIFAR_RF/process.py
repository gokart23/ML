import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import cPickle
import os

DATA_PATH = "data/"
FILE_PREFIX = "data_batch"
META_FILE = "batches.meta"
TEST_FILE = "test_batch"
label_names = []

def show_image(data):	
	# tmp = np.reshape(tmp, (32,32,3), order='F')
	tmp = data.reshape(3, 32, 32).transpose(1, 2, 0)
	plt.imshow(tmp)
	plt.show()

def get_label_names(fname):
	with open(fname, 'rb') as f:
		ln = cPickle.load(f)
		return ln['label_names']

def unpack_test(filename):
	with open(filename) as f:
		t = cPickle.load(f)
		return t['data'],t['labels']		
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
test, test_labels = unpack_test(DATA_PATH + TEST_FILE)
print train.shape, labels.shape
print "Initializing random forest"
rf = RandomForestClassifier(n_estimators=75, min_samples_leaf=50, n_jobs=6)
print "Fitting data"
rf.fit(train, labels)
print "Training complete"
# show_image(train[0,:])
print "Testing"
y_train = rf.predict(train)
y_test = rf.predict(test)
count_train, count = sum ( y_train == labels ), sum( y_test == test_labels )
train_accuracy = 100 * count_train / float(len(train))
accuracy = 100 * count / float(len(test)) 
print "Train Accuracy",train_accuracy,"Count",count_train
print "Accuracy",accuracy, "Count",count
# raw_input()
# for x in range(len(y_test)):
# 	print "Predicted",y_test[x],"(",label_names[int(y_test[x])],")","Ground",test_labels[x],"(",label_names[test_labels[x]],")"
# 	show_image(test[x,:])