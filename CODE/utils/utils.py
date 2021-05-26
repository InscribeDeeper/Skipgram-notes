#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE
    try:
        return np.divide(x, np.linalg.norm(x, ord=2, axis=1, keepdims=True)+ 1e-20)# 除以自身行的2-范数; 同时增加eps,避免分母为0
    except:
        return np.divide(x, np.linalg.norm(x, ord=2, keepdims=True)+ 1e-20) # The x input shape like (10,), the output will be the same: (10,)
    ### END YOUR CODE
    # return x

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE
    e = np.exp(x - np.max(x)) # shift invariant to avoid overflow
    # Try 保证输入输出 保持格式不变
    try:
        return np.divide(e,np.sum(e,axis=1,keepdims=True)) # The x input shape like (10,1), the output will be the same: (10,1)
    except:
        return np.divide(e,np.sum(e)) # The x input shape like (10,), the output will be the same: (10,)
    ### END YOUR CODE
    # return x

m,n = 2,4
Matrix = np.random.randint(0, 9, (m, n))
print("#### input ################# \n", Matrix)
print("#### softmax output ################# \n",softmax(Matrix))
print("#### output row summation ################# \n",np.sum(softmax(Matrix),axis=1, keepdims=True)) # 行求和, 概率为1

print("#### normalizeRows output ################# \n",normalizeRows(Matrix))



