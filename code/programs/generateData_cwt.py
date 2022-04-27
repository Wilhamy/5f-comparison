import numpy as np
import scipy.io
import scipy.signal
import mat73
from sklearn.model_selection import train_test_split
import os
from einops import rearrange, reduce, repeat

from torch import save
# also need to import the transformer

DATADIR = r'cwt'# Path to the data
OUTDIR = r'output'

DATAFILE = r'CWT_NoSpatial_5F-SubjectC-151204-5St-SGLHand.mat' # TODO: loop over files to run
OUTFILE = r'cwt_out.npy'
mat = mat73.loadmat(os.path.join(DATADIR, DATAFILE)) # dictionary
data_obj = mat['data']
examples = data_obj['examples'] # (N x Ceeg x num of frequencies x T)
examples = rearrange(examples, 'n c f t -> n t (c f)')
labels = data_obj['labels'] # (N,)
print(examples.shape)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(examples, labels, test_size=0.2, random_state=42)
save_data = {}
save_data["x_train"] = x_train
save_data["x_test"] = x_test
save_data["y_train"] = y_train
save_data["y_test"] = y_test

_, traincounts = np.unique(y_train, return_counts=True)
_, testcounts = np.unique(y_test, return_counts=True)
print('train:', traincounts)
print('test:', testcounts)

np.save(os.path.join(OUTDIR, OUTFILE), save_data)