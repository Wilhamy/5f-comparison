import os
import numpy as np

import torch
from torch import nn
from torch import Tensor
from sklearn.metrics import confusion_matrix

from Trans import *

import seaborn as sb

path_model = r'D:\thewi\Documents\UM\WN22\ML\Project\Datasets\ml-project\bestmodel\unaug' # path to model
path_data = r'.\output\original_data.npy'

total_data = np.load(path_data, allow_pickle=True)

train_data = total_data.item()['x_train'].transpose(0,2,1) # GROUP10; transposed to form n x Ceeg x T
train_labels = total_data.item()['y_train'] # GROUP10
test_data = total_data.item()['x_test'].transpose(0,2,1) #GROUP10; transposed to form n x Ceeg x T
test_labels = total_data.item()['y_test'] #GROUP10

# print(test_labels.shape)
# _, n_Ceeg, n_time_steps = train_data.shape # image dimensions
# num_classes = len(np.unique(train_labels)) # NOTE: assumes that 
# print(train_data.shape)

model = torch.load(os.path.join(path_model, 'model'))
# preprocess the test set based on statistics calculated on the train set
mu, sigma, W = Trans.preprocessing(None, train_data, train_labels)
X_v = np.expand_dims((test_data - mu) / sigma, axis=1)
X_v = np.einsum('abcd, ce -> abed', X_v, W) # TODO: consider einsum?


test_dataT = torch.cuda.FloatTensor(X_v)
Tok, Cls = model(test_dataT)

y_pred = torch.max(Cls, 1)[1]
# print(f"predicted labels: {y_pred}")
# print(f"test_label: {test_label}"
y_pred = y_pred.cpu().numpy().astype(int)
# print(y_pred)
# print('shapes:', test_labels.shape, y_pred.shape)
# print(test_labels-1)
# print(y_pred == test_labels-1)
bool_acc = y_pred == test_labels-1
acc = float(np.sum(bool_acc)) / float(test_labels.size)

print("accuracy:", acc)
cm = confusion_matrix(test_labels-1,y_pred)

sb.heatmap(cm, annot=True, fmt="d")

plt.show()