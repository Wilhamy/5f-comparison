import os
import numpy as np
import matplotlib.pyplot as plt
'''plot_curves.py - create the convergence curves for the training and test sets located in the datafile'''
datafile = os.path.join('.','plotting','params_acc_loss_RECENT.npy')
data = np.load(datafile, allow_pickle=True).item()

key = list(data.keys())[0] # only plot the first key's value
value = data[key]

train_accs = value['all_train_accs'][0]
val_accs = value['all_val_accs'][0]

train_loss = value['all_train_loss'][0]
val_loss = value['all_val_loss'][0]

# print(type(train_accs), type(val_accs), '\n', type(train_loss), type(val_loss))
# print(train_accs)
# # print(train_accs.shape, val_accs.shape, '\n', train_loss.shape, val_loss.shape)
# raise NotImplementedError
# plot accs on same plot
fig1, ax1 = plt.subplots()
ax1.plot(train_accs, label='Train')
ax1.plot(val_accs, label='Test')
ax1.set_xlabel("Epoch number (in 10s)")
ax1.set_ylabel("Acc")
ax1.set_title('Accuracy')
ax1.set_ybound([0,1])
ax1.grid(True)
ax1.legend()
# plot loss on same plot
fig2, ax2 = plt.subplots()
ax2.plot(train_loss, label='Train')
ax2.plot(val_loss, label='Test')
ax2.set_xlabel("Epoch number (in 10s)")
ax2.set_ylabel("Loss") # TODO: explain which loss?
ax2.set_title('Loss')
ax2.grid(True)
ax2.legend()

plt.show()