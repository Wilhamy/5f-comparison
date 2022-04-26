import os
import numpy as np

self.total_data = np.load(os.path.join(self.path,self.filename), allow_pickle=True)

self.train_data = self.total_data.item()['x_train'].transpose(0,2,1) # GROUP10; transposed to form n x Ceeg x T
# print(self.train_data.shape)

self.train_labels = self.total_data.item()['y_train'] # GROUP10
self.test_data = self.total_data.item()['x_test'].transpose(0,2,1) #GROUP10; transposed to form n x Ceeg x T
self.test_labels = self.total_data.item()['y_test'] #GROUP10
