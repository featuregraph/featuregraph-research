import numpy as np
from numpy.linalg import inv

#create absorbing Markov chain problem using numpy

arr = np.array([
  [0,1,0,0,0,1],  # s0, the initial state, goes to s1 and s5 with equal probability
  [4,0,0,3,2,0],  # s1 can become s0, s3, or s4, but with different probabilities
  [0,0,0,0,0,0],  # s2 is terminal and unreachable (never observed in practice)
  [0,0,0,0,0,0],  # s3 is terminal
  [0,0,0,0,0,0],  # s4 is terminal
  [0,0,0,0,0,0],  # s5 is terminal
])

arr2 = np.array([
    [0, 2, 1, 0, 0], 
    [0, 0, 0, 3, 4], 
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0]
])

#where does this converge? 

def absorbing_markov_chain(arr):
    idx = arr.sum(axis=1).argsort()
    arr = arr[:, idx][idx, :]
    idx = np.where(arr[:,:].sum(axis=1) == 0)

    arr[idx, idx] = 1
    len_idx = idx[0].shape[0]

    arr = arr/arr.sum(axis=1)[:,None]

    #get I, 0, R, Q from arr

    zero = arr[:len_idx, len_idx:]
    q = arr[len_idx:, len_idx:]
    r = arr[len_idx:, :len_idx]
    i = np.identity(len(q))
    f = inv(i - q)
    
    return f.dot(r)[0]
