import os
import numpy as np

params_accs_loss = np.load(r"D:\aditya\UMich\EECS545\Project\ml-project\output\params_acc_loss.npy", allow_pickle=True).item()
print(params_accs_loss.values())