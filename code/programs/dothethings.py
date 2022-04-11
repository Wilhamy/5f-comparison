import numpy as np
import torch
import torch.nn as nn
import scipy.io
import os
# also need to import the transformer

DATADIR = r'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\raw'# Path to the data
OUTDIR = r'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\output'


## TODO: START FOR HERE
DATAFILE = r'RAW_5F-SubjectC-151204-5St-SGLHand.mat' # TODO: loop over files to run
mat = scipy.io.loadmat(os.path.join(DATADIR, DATAFILE), struct_as_record=True) # dictionary
data_obj = mat['data'][0,0]
examples = data_obj[2]
labels = data_obj[3]
chnames = data_obj[4]

N,T,Ceeg = examples.shape # number of examples, number of time steps per ex, number of channels
examples = examples.transpose(0,2,1)

## Pre-processing
#bandpass ?

#z-score standardization



## TODO: END FOR HERE