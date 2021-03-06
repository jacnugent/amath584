"""
facial_recognition.py

Python code for Homework 2, AMATH 584, Fall 2020

Author: Jacqueline Nugent 
Last Modified: October 21, 2020
"""
import cv2 
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image

""" 
Set to True if you would like to save the plots:
"""
SAVE = False


"""
Specify file paths:
"""
MAIN_DIR = '/Users/jmnugent/Documents/__Year_3_2020-2021/AMATH_584-Numerical_Linear_Algebra/Homework/python/'
CROP_DIR = MAIN_DIR + 'CroppedYale/'
UNCROP_DIR = MAIN_DIR + 'yalefaces_uncropped/yalefaces/'
SAVE_DIR = MAIN_DIR + 'amath584/hw2_SVD_facial_recognition/'


"""
Read in the data:
"""
# set to true to print examples of the data to check it
# was read in correctly
check_data = False

### Cropped: ###
# get a list of paths to each subfolder in CroppedYale
paths = [CROP_DIR + dirname for dirname in os.listdir(CROP_DIR) if os.path.isdir(os.path.join(CROP_DIR, dirname))]

# initialize list to hold the averaged data matrices for each image
n_img = len(paths)
cropped_pics = [[]]*n_img
cropped_avgs = [[]]*n_img

for i in range(n_img):
    # get the list of file names within the subfolder for that image
    subfolder = paths[i] + '/'
    imagenames = [subfolder + f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
    
    # make one list containing the data matrices for each (grayscale) image 
    cropped_pics[i] = [cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2GRAY) for x in imagenames]

    # averaged the data matrix for this image and add to the list
    cropped_avgs[i] = np.mean(cropped_pics[i], axis=0)
    
if check_data:
    for i in range(len(cropped_avgs)):
        plt.imshow(cropped_avgs[i], cmap='gray')
        plt.axis('off')
        plt.show()
    for i in range(0, 10):
        for j in range(0, 3):
            plt.imshow(cropped_pics[i][j], cmap='gray')
            plt.axis('off')
        plt.show()
    
### Uncropped: ###
# get a list of paths to each subfolder in yalefaces_uncropped
unpaths = sorted([UNCROP_DIR + name for name in os.listdir(UNCROP_DIR)])

# check that each subject has 11 images in the dataset
if len(unpaths) % 11 == 0:
    n_sub = int(len(unpaths) / 11)
else:
    raise Exception('Invalid number of images! Each subject should have exactly 11 images.')

# initialize list to hold the averaged data matrices for each subject
uncropped_pics = [[]]*n_sub
uncropped_avgs = [[]]*n_sub

for i in range(n_sub):
    # make one list containing the data matrices of the 11 images for that subject
    uncropped_pics[i] = [np.array(Image.open(x).convert('L')) for x in unpaths[i*11:(i+1)*11]]

    # averaged the data matrix for this image and add to the list
    uncropped_avgs[i] = np.mean(uncropped_pics[i], axis=0)

if check_data:
    for i in range(len(uncropped_avgs)):
        plt.imshow(uncropped_avgs[i], cmap='gray')
        plt.axis('off')
        plt.show()
    for i in range(0, 3):
        for j in range(len(uncropped_pics[i])):
            plt.imshow(uncropped_pics[i][j], cmap='gray')
            plt.axis('off')
            plt.show()

            
"""
Split into training and testing data (use ~70% of the imaages for
each subject to train):
"""
### Cropped: ###
n_train_c = int(np.round(len(cropped_pics[0])*0.7))

crop_train = [x[:n_train_c] for x in cropped_pics]
crop_test = [x[n_train_c:] for x in cropped_pics]

crop_train_avgs = [np.mean(x, axis=0) for x in crop_train]
crop_test_avgs = [np.mean(x, axis=0) for x in crop_test]

### Uncropped: ###
n_train_uc = int(np.round(len(uncropped_pics[0])*0.7))

uncrop_train = [x[:n_train_uc] for x in uncropped_pics]
uncrop_test = [x[n_train_uc:] for x in uncropped_pics]

uncrop_train_avgs = [np.mean(x, axis=0) for x in uncrop_train]
uncrop_test_avgs = [np.mean(x, axis=0) for x in uncrop_test]


"""
1. Do an SVD analysis of the images
"""
### Cropped: ###
# stack so each image is one column in the data matrix
all_pics_c = [cropped_pics[i][j].flatten() for i in range(len(crop_train)) for j in range(len(crop_train[i]))]
A = np.transpose(np.asarray(all_pics_c))

# perform (economy) SVD
[U, S, VT] = np.linalg.svd(A, full_matrices=False)


### Uncropped: ###
# stack so each image is one column in the data matrix
all_pics_uc = [uncropped_pics[i][j].flatten() for i in range(len(uncrop_train)) for j in range(len(uncrop_train[i]))]
A_uc = np.transpose(np.asarray(all_pics_uc))

# perform (economy) SVD
[U_uc, S_uc, VT_uc] = np.linalg.svd(A_uc, full_matrices=False)

# check that the dimensions are correct
print(A_uc.shape, U_uc.shape, S_uc.shape, VT_uc.shape)


"""
2. What is the interpretation of the matrices? (plot the first few
   reshaped columns of U)
"""
### Cropped: ###
# get dimensions from the first pic
m, n = crop_train[0][0].shape

# get the eigenfaces (vectors) and reshape
eigenfaces = [np.reshape(U[:, i], (m, n)) for i in range(U.shape[1])]

# plot:
nfaces = 4

fig, axes = plt.subplots(1, nfaces, figsize=(6*nfaces, 6))

for i in range(len(axes)):
    axes[i].imshow(eigenfaces[i], cmap='gray')
    
    # label which eigenface it is
    if i == 0:
        lab = '1st'
    elif i == 1:
        lab = '2nd'
    elif i == 2:
        lab = '3rd'
    else:
        lab = '{}th'.format(i+1)
    axes[i].set_title(lab + ' Eigenface', fontsize=15)
    
    axes[i].axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_cropped_first_{}_eigenfaces.png'.format(nfaces), dpi=300, bbox_inches='tight')

plt.show()

### Uncropped: ###
# get dimensions from the first pic
mm, nn = uncrop_train[0][0].shape

# get the eigenfaces (vectors) and reshape
eigenfaces_uc = [np.reshape(U_uc[:, i], (mm, nn)) for i in range(U_uc.shape[1])]

# plot:
nfaces = 4

fig, axes = plt.subplots(1, nfaces, figsize=(6*nfaces, 6))

for i in range(len(axes)):
    axes[i].imshow(eigenfaces_uc[i], cmap='gray')
    
    # label which eigenface it is
    if i == 0:
        lab = '1st'
    elif i == 1:
        lab = '2nd'
    elif i == 2:
        lab = '3rd'
    else:
        lab = '{}th'.format(i+1)
    axes[i].set_title(lab + ' Eigenface', fontsize=15)
    
    axes[i].axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_uncropped_first_{}_eigenfaces.png'.format(nfaces), dpi=300, bbox_inches='tight')
plt.show()


"""
3. What does the singular value spectrum look like and how many modes are
   necessary for good image reconstructions using the PCA basis?
   (i.e. what is the rank r of the face space?)
"""
##### SVD spectrum: #####

### Cropped: ###
# plot all singular values:
nbig_c = 50
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121)
ax.scatter(np.arange(0, len(S)), S, color='C0', marker='o')
ax.set_yscale('log')
ax.set_ylim((1e-13, 1e7))
ax.tick_params(axis='both', labelsize=14)
ax.set_ylabel('Singular values, $\sigma_k$', fontsize=16)
ax.set_xlabel('$k$', fontsize=16)
ax.set_title('All Singular Values', fontsize=16)

ax2 = fig.add_subplot(122)
ax2.scatter(np.arange(0, nbig_c), S[:nbig_c], color='C0', marker='o')
ax2.set_yscale('log')
ax2.set_ylim((1e4, 1e6))
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel('Singular values, $\sigma_k$', fontsize=16)
ax2.set_xlabel('$k$', fontsize=16)
ax2.set_title('First {} Singular Values'.format(nbig_c), fontsize=16)

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_cropped_singular_values.png'.format(nbig_c), dpi=300, bbox_inches='tight')
plt.show()


# bonus plot: project all 38 people onto the first 50 modes 
nbig_c = 50
projs = [np.matmul(np.reshape(x, (1, m*n)), U[:, :nbig_c]).flatten() 
         for x in crop_train_avgs]

nrow = 4
ncol = 10
tsize = 30
lsize = 25

fig, axes = plt.subplots(nrow, ncol, figsize=(8*ncol, 6*nrow))
plt.subplots_adjust(hspace=0.4)

for r in range(nrow):
    for c in range(ncol):
        i = c + (ncol)*r
        if i < len(crop_train_avgs):
            axes[r, c].bar(np.arange(1, nbig_c), projs[i][1:])
            if i < 10:
                axes[r, c].set_title('yaleB0{}'.format(i+1), fontsize=tsize)
            elif i > 12: # since B14 (i=13) is skipped
                axes[r, c].set_title('yaleB{}'.format(i+2), fontsize=tsize)
            else:
                axes[r, c].set_title('yaleB{}'.format(i+1), fontsize=tsize)
                
            axes[r, c].set_ylim((-1200, 1200))
            
            # only label y axis for the first column:
            if c == 0:
                axes[r, c].tick_params(axis='y', labelsize=lsize)
                axes[r, c].set_xticklabels([])

            else:
                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
        else:
            axes[r, c].axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_cropped_ALL_proj_onto_2-{}_eigfaces.png'.format(nbig_c),
                dpi=300, bbox_inches='tight')

plt.show()



### Uncropped: ###
# Plot all singular values:
nbig_uc = 111
fig = plt.figure(figsize=(14, 6))

ax = fig.add_subplot(121)
ax.scatter(np.arange(0, len(S_uc)), S_uc, color='C0', marker='o')
ax.set_yscale('log')
ax.set_ylim((1e-13, 1e7))
ax.tick_params(axis='both', labelsize=14)
ax.set_ylabel('Singular values, $\sigma_k$', fontsize=16)
ax.set_xlabel('$k$', fontsize=16)
ax.set_title('All Singular Values', fontsize=16)

ax2 = fig.add_subplot(122)
ax2.scatter(np.arange(0, nbig_uc), S_uc[:nbig_uc], color='C0', marker='o')
ax2.set_yscale('log')
ax2.set_ylim((1e3, 1e6))
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel('Singular values, $\sigma_k$', fontsize=16)
ax2.set_xlabel('$k$', fontsize=16)
ax2.set_title('First {} Singular Values'.format(nbig_uc), fontsize=16)

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_uncropped_singular_values.png'.format(nbig_uc), dpi=300, bbox_inches='tight')
plt.show()

# bonus plot: project all 15 people onto the first 20 modes
nbig_uc = 20
projs_uc = [np.matmul(np.reshape(x, (1, mm*nn)), U_uc[:, :nbig_uc]).flatten() for x in uncrop_train_avgs]

nrow = 3
ncol = 5
tsize = 30
lsize = 25

fig, axes = plt.subplots(nrow, ncol, figsize=(8*ncol, 6*nrow))

for r in range(nrow):
    for c in range(ncol):
        i = c + (ncol)*r
        if i < len(uncrop_train_avgs):
            axes[r, c].bar(np.arange(1, nbig_uc), projs_uc[i][1:])
            if i < 9:
                axes[r, c].set_title('subject0{}'.format(i+1), fontsize=tsize)
            else:
                axes[r, c].set_title('subject{}'.format(i+1), fontsize=tsize)

            axes[r, c].set_ylim((-12000, 12000))
              
            # only label y axis for the first column:
            if c == 0:
                axes[r, c].tick_params(axis='both', labelsize=lsize)
                axes[r, c].set_xticklabels([])

            else:
                axes[r, c].set_xticklabels([])
                axes[r, c].set_yticklabels([])
        else:
            axes[r, c].axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yale_uncropped_ALL_proj_onto_2-{}_eigfaces.png'.format(nbig_uc), dpi=300, bbox_inches='tight')
plt.show()


##### rank of the face space: #####
# set the threshold to be 99.99%; after some testing, this was sufficient to get
# reconstructions very close to the original image in both the cropped and
# uncropped data

### Cropped: ###
S_sq = S**2
threshold = 0.97

# equation: sum of first t eigenvalues divided by sum of all eigenvalues >= threshold
for t in np.arange(0, len(S)):
    pct = np.sum(S[:t]**2)/np.sum(S**2)
    if pct >= threshold:
        print('{t} modes has fraction >= threshold of {th}%'.format(t=t, th=threshold*100.))
        nmode_c = t
        break

### Uncropped: ###
S_sq_uc = S_uc**2
threshold = 0.99

# equation: sum of first t eigenvalues divided by sum of all eigenvalues >= threshold
for t in np.arange(0, len(S_uc)):
    pct_uc = np.sum(S_uc[:t]**2)/np.sum(S_uc**2)
    if pct_uc >= threshold:
        print('{t} modes has pct {p}%'.format(t=t, p=pct_uc*100.))
        nmode_uc = t
        break      


"""
4. Compare the difference between the cropped (and aligned) vs. uncropped
   images in terms of singular value decay and reconstruction capabilities.   
"""
### Cropped reconstructions: ###
## reconstruct some of the training data: (choose arbitrarily) ##
# B01:
vec1 = np.reshape(crop_train[0][3], (1, 32256))
proj1 = np.matmul(vec1, U)
proj1_ef = np.matmul(vec1, U[:, :nmode_c])
recon1_all = np.matmul(proj1, np.transpose(U))
recon1_all_rs = np.reshape(recon1_all, (192, 168))
recon1_ef = np.matmul(proj1_ef, np.transpose(U[:, :nmode_c]))
recon1_ef_rs = np.reshape(recon1_ef, (192, 168))

# B02:
vec2 = np.reshape(crop_train[1][3], (1, 32256))
proj2 = np.matmul(vec2, U)
proj2_ef = np.matmul(vec2, U[:, :nmode_c])
recon2_all = np.matmul(proj2, np.transpose(U))
recon2_all_rs = np.reshape(recon2_all, (192, 168))
recon2_ef = np.matmul(proj2_ef, np.transpose(U[:, :nmode_c]))
recon2_ef_rs = np.reshape(recon2_ef, (192, 168))

# plot:
tsize = 25
fig = plt.figure(figsize=(16, 16))

ax1 = fig.add_subplot(221)
ax1.imshow(crop_train[0][3], cmap='gray')
ax1.set_title('yaleB01 - original\n ', fontsize=tsize)
ax1.axis('off')

ax3 = fig.add_subplot(222)
ax3.imshow(recon1_ef_rs, cmap='gray')
ax3.set_title('yaleB01 - reconstructed\n({} eigenfaces)'.format(nmode_c), fontsize=tsize)
ax3.axis('off')

ax4 = fig.add_subplot(223)
ax4.imshow(crop_train[1][3], cmap='gray')
ax4.set_title('yaleB02 - original\n ', fontsize=tsize)
ax4.axis('off')

ax6 = fig.add_subplot(224)
ax6.imshow(recon2_ef_rs, cmap='gray')
ax6.set_title('yaleB02 - reconstructed\n({} eigenfaces)'.format(nmode_c), fontsize=tsize)
ax6.axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yaleB01-02_reconstructed_{}_modes.png'.format(nmode_c), dpi=300, bbox_inches='tight')
plt.show()


## reconstruct some of the testing data: (choose arbitrarily) ##
# B37:
vec1_test = np.reshape(crop_test[-2][-1], (1, 32256))
proj1_test = np.matmul(vec1_test, U)
proj1_ef_test = np.matmul(vec1_test, U[:, :nmode_c])
recon1_all_test = np.matmul(proj1_test, np.transpose(U))
recon1_all_rs_test = np.reshape(recon1_all_test, (192, 168))
recon1_ef_test = np.matmul(proj1_ef_test, np.transpose(U[:, :nmode_c]))
recon1_ef_rs_test = np.reshape(recon1_ef_test, (192, 168))

# B38:
vec2_test = np.reshape(crop_test[-1][-1], (1, 32256))
proj2_test = np.matmul(vec2_test, U)
proj2_ef_test = np.matmul(vec2_test, U[:, :nmode_c])
recon2_all_test = np.matmul(proj2_test, np.transpose(U))
recon2_all_rs_test = np.reshape(recon2_all_test, (192, 168))
recon2_ef_test = np.matmul(proj2_ef_test, np.transpose(U[:, :nmode_c]))
recon2_ef_rs_test = np.reshape(recon2_ef_test, (192, 168))

# plot:
tsize = 25
fig = plt.figure(figsize=(16, 16))

ax1 = fig.add_subplot(221)
ax1.imshow(crop_test[-2][-1], cmap='gray')
ax1.set_title('yaleB37 - original\n ', fontsize=tsize)
ax1.axis('off')

ax3 = fig.add_subplot(222)
ax3.imshow(recon1_ef_rs_test, cmap='gray')
ax3.set_title('yaleB37 - reconstructed\n({} eigenfaces)'.format(nmode_c), fontsize=tsize)
ax3.axis('off')

ax4 = fig.add_subplot(223)
ax4.imshow(crop_test[-1][-1], cmap='gray')
ax4.set_title('yaleB38 - original\n ', fontsize=tsize)
ax4.axis('off')

ax6 = fig.add_subplot(224)
ax6.imshow(recon2_ef_rs_test, cmap='gray')
ax6.set_title('yaleB38 - reconstructed\n({} eigenfaces)'.format(nmode_c), fontsize=tsize)
ax6.axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'yaleB37-38_reconstructed_{}_modes.png'.format(nmode_c), dpi=300, bbox_inches='tight')
plt.show()


### Uncropped reconstructions: ###
## reconstruct some of the training data: (choose arbitrarily) ##
# subject01:
vec1_uc = np.reshape(uncrop_train[0][0], (1, 77760))
proj1_uc = np.matmul(vec1_uc, U_uc)
proj1_ef_uc = np.matmul(vec1_uc, U_uc[:, :nmode_uc])
recon1_all_uc = np.matmul(proj1_uc, np.transpose(U_uc))
recon1_all_rs_uc = np.reshape(recon1_all_uc, (243, 320))
recon1_ef_uc = np.matmul(proj1_ef_uc, np.transpose(U_uc[:, :nmode_uc]))
recon1_ef_rs_uc = np.reshape(recon1_ef_uc, (243, 320))


# subject02:
vec2_uc = np.reshape(uncrop_train[1][0], (1, 77760))
proj2_uc = np.matmul(vec2_uc, U_uc)
proj2_ef_uc = np.matmul(vec2_uc, U_uc[:, :nmode_uc])
recon2_all_uc = np.matmul(proj2_uc, np.transpose(U_uc))
recon2_all_rs_uc = np.reshape(recon2_all_uc, (243, 320))
recon2_ef_uc = np.matmul(proj2_ef_uc, np.transpose(U_uc[:, :nmode_uc]))
recon2_ef_rs_uc = np.reshape(recon2_ef_uc, (243, 320))

# plot:
tsize = 20
fig = plt.figure(figsize=(16, 16))

ax1 = fig.add_subplot(221)
ax1.imshow(uncrop_train[0][0], cmap='gray')
ax1.set_title('subject01 - original\n ', fontsize=tsize)
ax1.axis('off')

ax4 = fig.add_subplot(222)
ax4.imshow(recon1_ef_rs_uc, cmap='gray')
ax4.set_title('subject01 - reconstructed\n({} eigenfaces)'.format(nmode_uc), fontsize=tsize)
ax4.axis('off')

ax5 = fig.add_subplot(223)
ax5.imshow(uncrop_train[1][0], cmap='gray')
ax5.set_title('subject02 - original\n ', fontsize=tsize)
ax5.axis('off')

ax6 = fig.add_subplot(224)
ax6.imshow(recon2_ef_rs_uc, cmap='gray')
ax6.set_title('subject02 - reconstructed\n({} eigenfaces)'.format(nmode_uc), fontsize=tsize)
ax6.axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'subject01-02_reconstructed_{}_modes.png'.format(nmode_uc), dpi=300, bbox_inches='tight')
plt.show()


## reconstruct some of the testing data: (choose arbitrarily) ##
# subject14:
vec1_uc_test = np.reshape(uncrop_test[-2][-1], (1, 77760))
proj1_uc_test = np.matmul(vec1_uc_test, U_uc)
proj1_ef_uc_test = np.matmul(vec1_uc_test, U_uc[:, :nmode_uc])
recon1_all_uc_test = np.matmul(proj1_uc_test, np.transpose(U_uc))
recon1_all_rs_uc_test = np.reshape(recon1_all_uc_test, (243, 320))
recon1_ef_uc_test = np.matmul(proj1_ef_uc_test, np.transpose(U_uc[:, :nmode_uc]))
recon1_ef_rs_uc_test = np.reshape(recon1_ef_uc_test, (243, 320))


# subject15:
vec2_uc_test = np.reshape(uncrop_test[-1][-1], (1, 77760))
proj2_uc_test = np.matmul(vec2_uc_test, U_uc)
proj2_ef_uc_test = np.matmul(vec2_uc_test, U_uc[:, :nmode_uc])
recon2_all_uc_test = np.matmul(proj2_uc_test, np.transpose(U_uc))
recon2_all_rs_uc_test = np.reshape(recon2_all_uc_test, (243, 320))
recon2_ef_uc_test = np.matmul(proj2_ef_uc_test, np.transpose(U_uc[:, :nmode_uc]))
recon2_ef_rs_uc_test = np.reshape(recon2_ef_uc_test, (243, 320))

# plot:
tsize = 20
fig = plt.figure(figsize=(16, 16))

ax1 = fig.add_subplot(221)
ax1.imshow(uncrop_test[-2][-1], cmap='gray')
ax1.set_title('subject14 - original\n ', fontsize=tsize)
ax1.axis('off')

ax3 = fig.add_subplot(222)
ax3.imshow(recon1_ef_rs_uc_test, cmap='gray')
ax3.set_title('subject14 - reconstructed\n({} eigenfaces)'.format(nmode_uc), fontsize=tsize)
ax3.axis('off')

ax5 = fig.add_subplot(223)
ax5.imshow(uncrop_test[-1][-1], cmap='gray')
ax5.set_title('subject15 - original\n ', fontsize=tsize)
ax5.axis('off')

ax7 = fig.add_subplot(224)
ax7.imshow(recon2_ef_rs_uc_test, cmap='gray')
ax7.set_title('subject15 - reconstructed\n({} eigenfaces)'.format(nmode_uc), fontsize=tsize)
ax7.axis('off')

if SAVE:
    plt.savefig(SAVE_DIR + 'subject14-15_reconstructed_{}.png'.format(nmode_uc), dpi=300, bbox_inches='tight')
plt.show()


### Plot singular value decay: ###
tsize = 20
fsize = 18
lsize = 16

fig = plt.figure(figsize=(18, 6))
plt.subplots_adjust(wspace=0.25)

ax1 = fig.add_subplot(121)
ax1.plot(np.arange(0, len(S)), S, color='b', label='Cropped')
ax1.scatter(np.arange(0, len(S)), S, s=10, color='b', marker='o', label='Cropped')
ax1.set_yscale('log')
ax1.set_ylabel('Singular values, $\sigma_k$', fontsize=fsize)
ax1.set_xlabel('$k$', fontsize=fsize)
ax1.set_ylim((1e-13, 1e7))

ax1.tick_params(axis='both', labelsize=lsize)
ax1.tick_params(axis='x', color='b', labelcolor='b')
ax1.set_title('All Singular Values\n ', fontsize=tsize)

ax2 = ax1.twiny()
ax2.plot(np.arange(0, len(S_uc)), S_uc, color='r', label='Uncropped')
ax2.scatter(np.arange(0, len(S_uc)), S_uc, s=10, color='r', marker='o', label='Uncropped')
ax2.set_yscale('log')
ax2.set_ylabel('Singular values, $\sigma_k$', fontsize=fsize)
ax2.tick_params(axis='both', labelsize=lsize)
ax2.tick_params(axis='x', color='r', labelcolor='r')

ax2 = fig.add_subplot(122)
ax2.plot(np.arange(0, nmode_c), S[:nmode_c], color='b', label='Cropped')
ax2.scatter(np.arange(0, nmode_c), S[:nmode_c], s=10, color='b', marker='o', label='Cropped')
ax2.set_yscale('log')
ax2.set_ylabel('Singular values, $\sigma_k$', fontsize=fsize)
ax2.set_xlabel('$k$', fontsize=fsize)
ax2.set_ylim((1e4, 1e6))
ax2.tick_params(axis='both', labelsize=lsize)
ax2.tick_params(axis='x', color='b', labelcolor='b')
ax2.set_title('Largest Singular Values\n ', fontsize=tsize)

ax3 = ax2.twiny()
ax3.plot(np.arange(0, nmode_uc), S_uc[:nmode_uc], color='r', label='Uncropped')
ax3.scatter(np.arange(0, nmode_uc), S_uc[:nmode_uc], s=10, color='r', marker='o', label='Uncropped')
ax3.set_yscale('log')
ax3.set_ylabel('Singular values, $\sigma_k$', fontsize=fsize)
ax3.tick_params(axis='both', labelsize=lsize)
ax3.tick_params(axis='x', color='r', labelcolor='r')

# legend
crop_patch = mpatches.Patch(color='b', label='Cropped ({})'.format(nmode_c))
uncrop_patch = mpatches.Patch(color='r', label='Uncropped ({})'.format(nmode_uc))
plt.legend(handles=[crop_patch, uncrop_patch], fontsize=lsize, loc='upper right')

if SAVE:
    plt.savefig(SAVE_DIR + 'cropped_vs_uncropped_singular_values-{c}_{uc}_modes.png'.format(c=nmode_c, uc=nmode_uc),
                dpi=300, bbox_inches='tight')
plt.show()

