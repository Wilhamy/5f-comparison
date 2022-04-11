from enum import unique
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
    print(f"N, Ceeg, T: {N, Ceeg, T}")
    m1 = trials - trials.sum(2,keepdims=1)/N
    cov_trials = np.einsum('ijk,ilk->ijl',m1,m1) /(N - 1)

    print("cov_trials.shape:", cov_trials.shape)

    # onehot_matrix = generate_onehot(Y, num_classes = 3)
    # print("onehot_matrix:", onehot_matrix)

    # Rs = np.zeros((5, Ceeg, Ceeg))
    classes = sorted(set(Y))
    W = np.zeros((5,4,Ceeg))
    for c in classes:
       
        this_X = cov_trials[Y==c,:,:]
        R1 = np.mean(this_X, axis=0)

       
        other_X = cov_trials[Y!=c,:,:]
        R2 = np.mean(other_X,axis=0)
        
        R = R1+R2
        
        lamb, U = np.linalg.eig(R) # eigen decomp of R1+R2
        
        P = np.sqrt(np.diag(1/lamb))@U.T  # P = sqrt(lambda^-1) . U^T

        S1 = P@(R1@P.T)

        lam_s, B = np.linalg.eig(S1) # eigen decomp of S1

        lam_s_p = 1-lam_s # get lambda_s_prime
        s_idx = np.argsort(np.abs(lam_s_p))[::-1] #sort it in decreasing order

        # We know lambda_s_prime = (B^T P)R1(P^T B)
        # since filter is B^T P, sort it by decreasing order of magnitude of lambda_s_prime
        # take the first 4 values
        filter = B.T @ P 
        filter = filter[s_idx][:4]
        # add the filter fo
        W[c] = filter
    W = W.reshape((5*4, Ceeg))
    print(f"W shape: {W.shape}")
        

         


trial = np.random.rand(10,21,170)
Y = np.array([1, 2,2,3, 4, 5,1,3,2,4]) - 1
# print("trial:", trial)

spatial_filter(trial, Y)


