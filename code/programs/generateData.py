import numpy as np
import scipy.io
import scipy.signal
from sklearn.model_selection import train_test_split
import os

from torch import save
# also need to import the transformer

DATADIR = r'..\..\raw'# Path to the data
OUTDIR = r'..\..\output'


## TODO: START FOR HERE
DATAFILE = r'RAW_5F-SubjectC-151204-5St-SGLHand.mat' # TODO: loop over files to run
mat = scipy.io.loadmat(os.path.join(DATADIR, DATAFILE), struct_as_record=True) # dictionary
data_obj = mat['data'][0,0]
examples = data_obj[2]
labels = data_obj[3].flatten()
# chnames = data_obj[4]
N,T,Ceeg = examples.shape # number of examples, number of time steps per ex, number of channels
# examples = examples.transpose(0,2,1) # transpose is being done in trans.py already

## Pre-processing
#bandpass ?
fc = 200 # sample freq

Wl = 4
Wh = 40

Wn = np.array([Wl*2, Wh*2])/fc
b,a = scipy.signal.cheby2(6,60, Wn, btype='bandpass')
for j in range(examples.shape[2]):
    examples[:,:,j] = scipy.signal.filtfilt(b,a,examples[:,:,j])

#z-score standardization
# this part seems to be already done in the data_loader in Trans.py

# train test split
x_train, x_test, y_train, y_test = train_test_split(examples, labels, test_size=0.33, random_state=42)
save_data = {}
save_data["x_train"] = x_train
save_data["x_test"] = x_test
save_data["y_train"] = y_train
save_data["y_test"] = y_test

np.save("..\..\output\saved_data.npy", save_data)