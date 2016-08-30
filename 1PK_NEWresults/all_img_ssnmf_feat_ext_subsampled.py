from loaddata import loaddata, get_train_data
from ssnmf_func import ssnmf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image as pimg
import matplotlib
import os
from sklearn.cluster import DBSCAN
from spectral import *
from sklearn import svm
import time
from sklearn.semi_supervised import label_propagation
from load_ssnmf_feat import ssnmf_feat




start_time = time.time()

color_val =spy_colors
color_val[0] = color_val[1]
color_val[1] = color_val[2]
color_val[2] = color_val[3]
color_val[3] = [0,0,0]
rng = np.random.RandomState(2345)
#train_images = [1,3,5,7,9,11,13,15,17,19]               this is the actual dataset , it was commented as we did not have time to process so much chunk of data
#test_images = [2,4,6,8,10,12,14,16,18,20]		this is the test dataset which was commented as we did not have so much time . pLease remove both the # for actual work 
train_images = [1]
test_images = [2]

rank = 10
relax_label = True

for noi,i in enumerate(train_images):
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print tot_labels.shape
    print np.histogram(tot_labels,range(6))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0 or tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    labeled_pos_tr = non_bg_pos + bg_pos[:20000]
    
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    print np.histogram(tot_labels[shuff_labeled_pos],[0,1,2,3,4,5])
    if noi == 0:
        data_train = data_oi[:,shuff_labeled_pos]
        label_train = tot_labels[shuff_labeled_pos]
    else:
        data_train = np.hstack((data_train,data_oi[:,shuff_labeled_pos]))
        label_train = np.hstack((label_train,tot_labels[shuff_labeled_pos]))
        
for noi,i in enumerate(test_images):
    data_pi = loaddata(i,0.07)
    data_oi = data_pi[0]
    tot_labels = data_pi[1]
    print tot_labels.shape
    print np.histogram(tot_labels,range(6))
    labeled_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0]
    unlabeled_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] == 0]
    bg_pos = [x for x in range(tot_labels.shape[0]) if tot_labels[x] == 4]
    non_bg_pos =[x for x in range(tot_labels.shape[0]) if tot_labels[x] != 0 and tot_labels[x]!=4]
    rng.shuffle(bg_pos)
    labeled_pos_tr = non_bg_pos + bg_pos[:10000]
    
    
    shuff_labeled_pos = rng.permutation(labeled_pos_tr)
    if noi == 0:
        data_test = data_oi[:,shuff_labeled_pos]
        label_test = tot_labels[shuff_labeled_pos]
    else:
        data_test = np.hstack((data_test,data_oi[:,shuff_labeled_pos]))
        label_test = np.hstack((label_test,tot_labels[shuff_labeled_pos]))

shufftr = range(data_train.shape[1])
shuffte = range(data_test.shape[1])
rng.shuffle(shufftr)
rng.shuffle(shuffte)


data_train = data_train[:,shufftr]
label_train = label_train[shufftr]
data_test = data_test[:,shuffte]
label_test = label_test[shuffte]

fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_subsampled/' 
 
if(os.path.exists(fullfile) == False):
    os.makedirs(fullfile)



f = fullfile + ('data_tr10_te10_1.npz')

np.savez(f,data_train=data_train,label_train=label_train,data_test=data_test,label_test=label_test)


print data_train.shape, label_train.shape, data_test.shape,label_test.shape
print np.histogram(label_train,range(6)),'\n'
print np.histogram(label_test,range(6))


ssnmf_input_data = np.hstack((data_train,data_test))
print ssnmf_input_data.shape
ssnmf_input_label = np.hstack((label_train,np.zeros_like(label_test)))
print ssnmf_input_label.shape
tot_labels =  np.hstack((label_train,label_test))
print tot_labels.shape



test_pos = [x for x in range(ssnmf_input_label.shape[0]) if (ssnmf_input_label[x] == 0  and tot_labels[x] !=0)]
print len(test_pos)
rank = 10
#l = 1					sunit commented this out on 23rd august
L_param = 0.001
					
iteration_numbers = [0,1,2,3,4,5,6,7,8,9,10,11]	     #the following line was added by sunit on 23rd august
list_of_lambdas_used = [0.1,0.5,1,2,3,4,5,6,7,8,9,10]    #the following line was added by sunit on 23rd august
data_reconstruction_error=[0,0,0,0,0,0,0,0,0,0,0,0]      #the following line was added by sunit on 23rd august                       
label_reconstruction_error=[0,0,0,0,0,0,0,0,0,0,0,0]      #the following line was added by sunit on 23rd august                    This kind  of initiialization ought not be done I shall change it with suitable ones
evaluation=[0,0,0,0,0,0,0,0,0,0,0,0]                        #the following line was added by sunit on 23rd august

for iter_num in iteration_numbers:           #the following line was added by sunit on 23rd august
	l = list_of_lambdas_used[iter_num]   #the following line was added by sunit on 23rd august
	feat_mat,label_mat,data_recon_err,label_recon_err,eval_s = ssnmf(ssnmf_input_data,ssnmf_input_label,rank,l,relax_label = True,L_param=L_param)  #the following line was added by sunit on 23rd august
	data_reconstruction_error[iter_num] = data_recon_err    #the following line was added by sunit on 23rd august to store for further evaluation
	label_reconstruction_error[iter_num] = label_recon_err  #the following line was added by sunit on 23rd august to store for further evaluation
	evaluation[iter_num] = eval_s                           #the following line was added by sunit on 23rd august to store for further evaluation

# printing the error matrices and the evaluation matrix that was obtained for analysis
for iter_num in iteration_numbers:           #the following line was added by sunit on 23rd august
	print("\n\n\n\n\n lets start evaluation \n\n\nd_r_e     l_r_e     ev"+data_reconstruction_error[iter_num]+"    "+label_reconstruction_error[iter_num]+"    "+evaluation[iter_num]+"       \n\n")   # this was added to print the evaluated error matrices for evaluation it was added by sunit on 23rd august 

#feat_mat,label_mat,data_recon_err,label_recon_err,eval_s = ssnmf(ssnmf_input_data,ssnmf_input_label,rank,l,relax_label = True,L_param=L_param)         this was commented by me sunit on 23rd august

fullfile = os.path.dirname(os.path.abspath(__file__))+ '/all_images_feat_ext_subsampled/' 
 
if(os.path.exists(fullfile) == False):
    os.makedirs(fullfile)
f = fullfile + ('all_img1010_feat_ssnmf_subsampled.npz')


np.savez(f,features =feat_mat,label_mat = label_mat,tot_labels=tot_labels, input_labels = ssnmf_input_label ,
         test_pos = test_pos,data_recon_err = data_recon_err,label_recon_err=label_recon_err)    


print("--- %s minutes ---" % ((time.time() - start_time)/60.))




