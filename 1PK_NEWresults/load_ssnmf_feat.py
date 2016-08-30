import numpy as np
import os

def ssnmf_feat(i):
    
    fullfile = os.path.dirname(os.path.abspath(__file__))+ ('/ssnmf_features_10/') 
    f = fullfile + ('ssnmf_feature_%d.npz'%(i))

    feat = np.load(f)
    return feat['features'],feat['target'], feat['s0'],feat['s1']
