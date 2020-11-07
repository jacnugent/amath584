"""
lu_pivot.py

Computes the LU Decomposition of a matrix with 
pivot. For Problem 5 on AMATH 584 midterm.

Author: Jacqueline Nugent
Last Modified: November 6, 2020
"""
import numpy as np


def lu_pivot(A):
    """
    Compute the LU decomposition of matrix A with pivot, i.e. PA = LU
    
    Args: 
        A (numpy array), a square matrix
    Returns: 
        P, L, U (numpy arrays), matrices of the same size as A
    """
    # check that the input matrix is square
    m, n = np.shape(A)
    if m != n:
        raise Exception('Input matrix A must be square! Current size {m}x{n}.'.format(m=m, n=n))

    # initialize U as A and L and P as I
    U = A.copy()
    L = np.eye(m)
    P = np.eye(m)

    # loop through the first (m-1) columns:
    for k in range(0, m-1):
        
        # find max element in the column (the pivot)
        i = np.argmax(abs(U[k:, k])) + k
        
        # swap rows
        Pk = np.eye(m)
        Pk[[k, i], k:] = Pk[[i, k], k:]
        U[[k, i], k:] = U[[i, k], k:]
        L[[k, i], :k] = L[[i, k], :k]

        # now construct the subdiagonal elements of L and multiply
        for j in range(k+1, m):
            L[j, k] = U[j, k]/U[k, k]
            U[j, k:] = U[j, k:] - L[j, k]*U[k, k:]
        
        # multiply P by Pk to update
        P = Pk@P

    return [P, L, U]
