import numpy as np


def generate_onehot(labels, num_classes):
    '''generate_onehot(A, d) - turns vector A into a onehot matrix.
    Inputs:
        labels - the vector of shape (m,1)
        num_classes - the number of entries per onehot vector
    Outputs:
        H - the onehot matrix
    '''
    assert num_classes > labels.max()
    return (np.arange(num_classes) == labels[...,None]).astype(int)

#inputs:
#   Trials: N x Ceeg x T
#   Y Labels
def spatial_filter(trials, Y):
    N, Ceeg, T = trials.shape
    m1 = trials - trials.sum(2,keepdims=1)/N
    cov_trials = np.einsum('ijk,ilk->ijl',m1,m1) /(N - 1)

    print("cov_trials.shape:", cov_trials.shape)

    onehot_matrix = generate_onehot(Y, num_classes = 3)
    print("onehot_matrix:", onehot_matrix)

    



    pass




trial = (np.arange(24)).reshape((2,3,4))
Y = np.array([1, 2]) - 1
# print("trial:", trial)

spatial_filter(trial, Y)



