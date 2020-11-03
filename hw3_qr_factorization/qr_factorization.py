"""
qr_factorization.py

Python code for Homework 3, AMATH 584, Fall 2020

Author: Jacqueline Nugent 
Last Modified: November 2, 2020
"""
import math
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy import io


"""
Global Variables:
"""
# Save the random matrices so you can import into matlab?
SAVE_MATS = True 

# Save the figures?
SAVE_FIGS = True 


"""
Specify file paths:
"""
main_dir = '/Users/jmnugent/Documents/__Year_3_2020-2021/AMATH_584-Numerical_Linear_Algebra/Homework/python/'
save_dir = main_dir + 'amath584/hw3_qr_factorization/'


"""
1. QR Decomposition
"""
## Gram-Schmidt algorithm ##

def gram_schmidt(A):
    """
    Perform QR decomposition on input matrix A using
    the modified Gram-Schmidt orthogonalization procedure;
    gives the reduced form of the QR factorization of A.

    Input: matrix A (numpy array)
    Returns: matrices Q, R (numpy arrays)                                       
    """
    m, n = np.shape(A)
    Q = np.zeros(np.shape(A))
    R = np.zeros((n, n))

    # initialize V from A - this will make
    # q1 = direction of a1
    V = A.copy()
    
    # iterate through each column
    for i in range(0, n):
        # normalize vi to get qi
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]

        # recursively project all the pieces out
        for j in range(i, n):
            # projection of jth vector onto the ith direction
            R[i, j] = np.conj(Q[:, i]).T @ V[:, j]
            
            # subtract the vector that's orthogonal to that projection
            V[:, j] = V[:, j] - R[i, j]*Q[:, i]


    return Q, R


## Define some functions to check the performance of each algorithm ##

def check_orthonormal(Q, rel_tol=1e-9, abs_tol=1e-9):
    """
    Quick check if a columns of a matrix Q form an 
    orthonormal basis.
    
    Prints statement on if Q is orthonormal.
    """
    orthonorm = True
    
    # check if all columns have norm 1 (normal):
    for i in range(np.shape(Q)[1]):
        dot = Q[:, i] @ Q[:, i-1]
        norm = np.linalg.norm(Q[:, i-1])
        if not math.isclose(norm, 1, rel_tol=rel_tol):
            print('Q is NOT orthonormal: for at least one column, norm = {n} =/= 1'.format(n=norm))
            orthonorm = False
            break
            
    # check Q^T * Q = I (orthogonal):
    prod = Q.T @ Q
    identity = True
    for i in range(np.shape(prod)[0]):
        for j in range(np.shape(prod)[1]):
            
            # check diagonals are 1
            if i == j:
                if not math.isclose(prod[i, j], 1, rel_tol=rel_tol):
                    identity = False
                    break
            
            # check off-diagonals are zero
            else:
                if not math.isclose(prod[i, j], 0, abs_tol=abs_tol):
                    identity = False    
                    break
       
    if not identity:
        orthonorm = False
        print('Q is NOT orthonormal: the product Q^T * Q =/= I')

    if orthonorm:
        print('Q is orthonormal')


def check_size(A, Q, R):
    """
    Check the size of matrices Q and R from the QR 
    factorization of A.
    
    Prints statement on if Q and R have correct sizes
    """
    m, n = np.shape(A)
    
    if np.shape(A) == np.shape(Q):
        qsize = 'Q is the correct size'
    else:
        qsize = 'Q is NOT the correct size'
        
    if np.shape(R) != (n, n):
        rsize = 'R is NOT the correct size'
    else:
        rsize = 'R is the correct size'
    
    print(qsize + ' and ' + rsize)
    

### Set up some matrices to test ###

# choose m and n arbitrarily
m = 70
n = 10

# square
A_sq = np.random.randn(m, m)
print('Matrix of size {s} with cond(A) = {c}'.format(s=A_sq.shape, c=np.round(np.linalg.cond(A_sq), 1)))

# tall and skinny #1
A_ts1 = np.random.randn(m, n)
print('Matrix of size {s} with cond(A) = {c}'.format(s=A_ts1.shape, c=np.round(np.linalg.cond(A_ts1), 1)))

# tall and skinny #2
A_ts2 = np.random.randn(m*10, n)
print('Matrix of size {s} with cond(A) = {c}'.format(s=A_ts2.shape, c=np.round(np.linalg.cond(A_ts2), 1)))

# ill-conditioned matrix
A_orig = np.random.randn(m, n)
A_ic = np.hstack((A_orig, np.reshape(A_orig[:, 0], (m, 1))))
cn = '{:e}'.format(np.linalg.cond(A_ic))
print('Matrix of size {s} with cond(A) = {c}'.format(s=A_ic.shape, c=cn))

# put them in a list
mat_list = [A_sq, A_ts1, A_ts2, A_ic]
mat_names = ['m{}'.format(x) for x in range(len(mat_list))]

# save so you can load into matlab
if SAVE_MATS:
    io.savemat(save_dir + 'random_matrices.mat', mdict=dict(zip(mat_names, mat_list)))
    

### test the modified Gram-Schmidt algorithm ###
print('modified Gram-Schmidt:\n')
my_alg = [[]]*len(mat_list)

for i in range(len(mat_list)):
    Q, R = gram_schmidt(mat_list[i])
    my_alg[i] = Q @ R
    
    check_size(mat_list[i], Q, R)
    print('condition number of QR:', np.linalg.cond(my_alg[i]))
    diff = my_alg[i] - mat_list[i]
    print('norm of A - QR:'.format(i), np.linalg.norm(diff))
    check_orthonormal(Q)
    print('\n')

    
### test the matlab algorithm ###
# save the matrices above, load them into matlab,
# and run the script run_qrfactor.m
    
### test the python qr algorithm ###
print('Python\'s QR algorithm:')
python_alg = [[]]*len(mat_list)

for i in range(len(mat_list)):
    Q, R = np.linalg.qr(mat_list[i])
    python_alg[i] = Q @ R
    diff = python_alg[i] - mat_list[i]
    check_size(mat_list[i], Q, R)
    print('norm of A - QR:'.format(i), np.linalg.norm(diff))
    print('condition number of QR:', np.linalg.cond(python_alg[i]))
    check_orthonormal(Q)
    print('\n')


"""
2. Polynomial
"""
# function to make plots easier
def plot_polynomial(func, x, ylim, title, filename, save_dir, 
                    save=False):
    """ Plot the input polynomial func(x) and save figure if requested.
    """
    y = func(x)
    
    plt.plot(x, func(x))
    plt.title(title)
    plt.ylim(ylim)
    
    if save:
        plt.savefig(save_dir + filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
# polynomials:
p_rhs = lambda x: x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
p_lhs = lambda x: (x-2)**9

# interval
dx = 0.001
xspan = np.arange(1.920, 2.080, dx)

# plots:
plot_polynomial(p_rhs, xspan, ylim=(-1.5e-10, 1.5e-10),
                title='Right-hand side of $p(x)$', filename='px_rhs.png',
                save_dir=save_dir, save=SAVE_FIGS)

plot_polynomial(p_lhs, xspan, ylim=(-1.5e-10, 1.5e-10),
                title='Left-hand side of $p(x)$', filename='px_lhs.png',
                save_dir=save_dir, save=SAVE_FIGS)


"""
3. Consider the conditioning of a matrix
"""
### Condition number as a function of matrix size ###
# condition number as a function of size
condition_num = lambda m, n: np.linalg.cond(np.random.randn(m, n))

# get some values for m and n
ms = [int(x) for x in np.arange(1, 200)]
ns = [int(x) for x in np.arange(1, 100)]

# find the condition numbers for m > n
cs = np.zeros((len(ms), len(ns)))

for i in range(len(ms)):
    for j in range(len(ns)):
        m = ms[i]
        n = ns[j]
        if m == n:
            cs[i, j] = np.nan
        else:
            cs[i, j] = condition_num(m, n)

            
## plot 1: vary both m and n ##
tsize = 15
fsize=13

fig, ax = plt.subplots(figsize=(7, 7))

pcm = ax.pcolormesh(ns, ms, cs, cmap='PuBuGn', norm=colors.LogNorm()) 
cb = fig.colorbar(pcm, ax=ax)
cb.ax.tick_params(labelsize=fsize)
pcm.set_clim((1, 1e4))

ax.set_ylim((0, 200))
ax.set_xlim((0, 100))
ax.tick_params(axis='both', labelsize=fsize)
ax.set_ylabel('m    ', rotation=0, fontsize=tsize)
ax.set_xlabel('n', fontsize=tsize)
ax.set_title('Condition number of a random m x n matrix\n', fontsize=tsize)

if SAVE_FIGS:
    plt.savefig(save_dir + 'condition_number_function.png', dpi=300, bbox_inches='tight')

plt.show()


## plot 2: condition numbers for various almost-square matrices ##
# condition numbers for (m+1)xm matrix
cs_psq = [condition_num(m+1, m) for m in ms]

# plot
plt.plot(ms, cs_psq)

plt.yscale('log')
plt.ylim(.9, 1e4)

plt.ylabel('Condition number', fontsize=fsize)
plt.xlabel('m', fontsize=fsize)
plt.title('Condition number of a random (m+1) x m matrix')

if SAVE_FIGS:
    plt.savefig(save_dir + 'cond_num_near_psq.png', dpi=300, bbox_inches='tight')

plt.show()


### for a fixed m and n, copy the first col and append to (n+1)th col ###
def det_cond(m, n, print_results=True):
    """
    Appends the first column of a random mxn matrix A
    as the (n+1)th column of A. Prints the appended 
    size, the condition number, and the determinant if
    requested. Returns the appended random matrix A.
    """
    A = np.random.randn(m, n)

    # first column of A
    col_n1 = np.reshape(A[:, 0], (m, 1))

    # append as the (n+1)th column of A
    A = np.hstack((A, col_n1))

    if print_results:
        print('Original size: {}'.format((m, n)))
        print('Appended size: {}'.format(np.shape(A)))
        print('Condition number: {}'.format(np.linalg.cond(A)))
        print('Determinant: {}'.format(np.linalg.det(A)))
        
    return A

# try it for a few values of m and n
# (let n = m-1 so that A is a square matrix after you add a column)
ms = [2, 5, 10, 15, 20, 25]
ns = [m-1 for m in ms]

for i in range(len(ms)):
    det_cond(ms[i], ns[i])
    print('\n')

    
### add noise to the appended column & see what happens to cond number ###
# condition number as a function of epsilon
def cond_eps(m, n, eps):
    """
    Returns the condition number for a random 
    mxn, m>n, matrix A and noise scale epsilon.
    """
    A = np.random.randn(m, n)
    cond_orig = np.linalg.cond(A)
    
    # first column of A
    col_n1 = np.reshape(A[:, 0], (m, 1))

    # append as the (n+1)th column of A
    A = np.hstack((A, col_n1))
    
    # add noise
    A[:, n] = A[:, n] + eps*np.random.randn(1, m) 
    
    return [cond_orig, np.linalg.cond(A)]


# look at some examples
m = 40
n = 10

eps = [10**float(exp) for exp in np.arange(-16, 1)]

# print out (old condition #) --> (condition # after noise):
print('random {m}x{n} matrix:'.format(m=m, n=n), '\n')
for i in range(len(eps)):
    co, ca = cond_eps(m, n, eps[i])
    print('epsilon = {e}:\n    {co} --> {ca}'.format(e=eps[i],
                                                     co=np.round(co, 1),
                                                     ca=np.round(ca, 1)))

    
## make a a plot of condition number vs. epsilon ##
# find condition numbers
noisy_conds = [cond_eps(m, n, e)[1] for e in eps]

# plot
fig, ax = plt.subplots()

ax.scatter(eps, noisy_conds)

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim((5e-17, 5))
ax.set_ylim((1, 1e17))

ax.tick_params(axis='both', labelsize=fsize)
ax.set_ylabel('Condition number', fontsize=fsize)
ax.set_xlabel('Epsilon', fontsize=fsize)
ax.set_title('Condition number as a function of epsilon\nfor a random {m}x{n} matrix'.format(m=m, n=n),
             fontsize=tsize-1)

if SAVE_FIGS:
    plt.savefig(save_dir + 'condn_func_of_eps.png', dpi=300, bbox_inches='tight')

plt.show()
