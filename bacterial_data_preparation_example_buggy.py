# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:39:03 2018

@author: Jintram
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


bac_image_path = 'C:/Users/Jintram/Temporary/ecoli_set1/'
bac_image_path_x = bac_image_path + 'x_images/'
bac_image_path_y = bac_image_path + 'y_images/'

bac_imgs = os.listdir(bac_image_path_x)


# %% just show one image

testbacimg_x = np.array(imread(bac_image_path_x + bac_imgs[1]))
testbacimg_y = np.array(imread(bac_image_path_y + bac_imgs[1]))

plt.figure()
plt.subplot(121)
plt.imshow(testbacimg_x, cmap='gray')
plt.subplot(122)
plt.imshow(testbacimg_y, cmap='gray')
plt.show()

# %% Now create one test situation

# Note that we want each category to be represented well

# Perhaps first crop the image a bit to not get too many uniform background pixels
# Note: indexing = row, column
maxrow, maxcol = np.shape(testbacimg_y) # size img
idx_nonzero = testbacimg_y.nonzero() # location colony
# get sides + margin
MARGIN=30
crop_row1   = np.max([np.min(idx_nonzero[0])-MARGIN,0]) 
crop_row2   = np.min([np.max(idx_nonzero[0])+MARGIN,maxrow])
crop_col1   = np.max([np.min(idx_nonzero[1])-MARGIN,0])
crop_col2   = np.min([np.max(idx_nonzero[1])+MARGIN,maxcol])

testbacimg_x_2 = testbacimg_x[ crop_row1:crop_row2, crop_col1:crop_col2]
testbacimg_y_2 = testbacimg_y[ crop_row1:crop_row2, crop_col1:crop_col2]
plt.figure()
plt.subplot(121)
plt.imshow(testbacimg_x_2, cmap='gray')
plt.subplot(122)
plt.imshow(testbacimg_y_2, cmap='gray')
plt.show()

# now find the locations of the three categories (excl. margin)
MARGIN_MASK=10 # we cannot select pixels to close to boundary
idxs_bg   = np.nonzero(testbacimg_y_2==0)
idxs_bac  = np.nonzero(testbacimg_y_2==1)
idxs_edge = np.nonzero(testbacimg_y_2==2)
idxs_bg_i = idxs_bg[0]
idxs_bg_j = idxs_bg[1]
idxs_bac_i = idxs_bac[0]
idxs_bac_j = idxs_bac[1]
idxs_edge_i = idxs_edge[0]
idxs_edge_j = idxs_edge[1]
# apply margins
idxs_bg_i_sel = idxs_bg_i[idxs_bg_i>MARGIN_MASK]
idxs_bg_i_sel = idxs_bg_i_sel[idxs_bg_i_sel<(maxrow-MARGIN_MASK)]
idxs_bg_j_sel = idxs_bg_j[idxs_bg_j>MARGIN_MASK]
idxs_bg_j_sel = idxs_bg_j_sel[idxs_bg_j_sel<(maxcol-MARGIN_MASK)]

idxs_bac_i_sel = idxs_bac_i[idxs_bac_i>MARGIN_MASK]
idxs_bac_i_sel = idxs_bac_i_sel[idxs_bac_i_sel<(maxrow-MARGIN_MASK)]
idxs_bac_j_sel = idxs_bac_j[idxs_bac_j>MARGIN_MASK]
idxs_bac_j_sel = idxs_bac_j_sel[idxs_bac_j_sel<(maxcol-MARGIN_MASK)]

idxs_edge_i_sel = idxs_edge_i[idxs_edge_i>MARGIN_MASK]
idxs_edge_i_sel = idxs_edge_i_sel[idxs_edge_i_sel<(maxrow-MARGIN_MASK)]
idxs_edge_j_sel = idxs_edge_j[idxs_edge_j>MARGIN_MASK]
idxs_edge_j_sel = idxs_edge_j_sel[idxs_edge_j_sel<(maxcol-MARGIN_MASK)]

# now select one of each + margins
#train_slice_bg_idxs_i = idxs_bg_i_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_bg_idxs_j = idxs_bg_j_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_bg = testbacimg_y_2[train_slice_bg_idxs_i,train_slice_bg_idxs_j]
x_train_slice_bg = testbacimg_x_2[(idxs_bg_i_sel[1]-MARGIN_MASK):(idxs_bg_i_sel[1]+MARGIN_MASK+1),
                                (idxs_bg_j_sel[1]-MARGIN_MASK):(idxs_bg_j_sel[1]+MARGIN_MASK+1)]
y_train_slice_bg = testbacimg_y_2[(idxs_bg_i_sel[1]-MARGIN_MASK):(idxs_bg_i_sel[1]+MARGIN_MASK+1),
                                (idxs_bg_j_sel[1]-MARGIN_MASK):(idxs_bg_j_sel[1]+MARGIN_MASK+1)]

#train_slice_bac_idxs_i = idxs_bac_i_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_bac_idxs_j = idxs_bac_j_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_bac = testbacimg_y_2[train_slice_bac_idxs_i,train_slice_bac_idxs_j]
x_train_slice_bac = testbacimg_x_2[(idxs_bac_i_sel[1]-MARGIN_MASK):(idxs_bac_i_sel[1]+MARGIN_MASK+1),
                                (idxs_bac_j_sel[1]-MARGIN_MASK):(idxs_bac_j_sel[1]+MARGIN_MASK+1)]
y_train_slice_bac = testbacimg_y_2[(idxs_bac_i_sel[1]-MARGIN_MASK):(idxs_bac_i_sel[1]+MARGIN_MASK+1),
                                (idxs_bac_j_sel[1]-MARGIN_MASK):(idxs_bac_j_sel[1]+MARGIN_MASK+1)]


#train_slice_edge_idxs_i = idxs_edge_i_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_edge_idxs_j = idxs_edge_j_sel[1]+list(range(-MARGIN_MASK,MARGIN_MASK+1))
#train_slice_edge = testbacimg_y_2[train_slice_edge_idxs_i,train_slice_edge_idxs_j]
x_train_slice_edge = testbacimg_x_2[(idxs_edge_i_sel[1]-MARGIN_MASK):(idxs_edge_i_sel[1]+MARGIN_MASK+1),
                                (idxs_edge_j_sel[1]-MARGIN_MASK):(idxs_edge_j_sel[1]+MARGIN_MASK+1)]
y_train_slice_edge = testbacimg_y_2[(idxs_edge_i_sel[1]-MARGIN_MASK):(idxs_edge_i_sel[1]+MARGIN_MASK+1),
                                (idxs_edge_j_sel[1]-MARGIN_MASK):(idxs_edge_j_sel[1]+MARGIN_MASK+1)]


# now show what we're doing

plt.figure()
plt.subplot(231)
plt.title('background')
plt.imshow(x_train_slice_bg, cmap='gray')
plt.subplot(232)
plt.title('bacterium')
plt.imshow(x_train_slice_bac, cmap='gray')
plt.subplot(233)
plt.title('edge')
plt.imshow(x_train_slice_edge, cmap='gray')

plt.subplot(234)
plt.imshow(y_train_slice_bg, cmap='gray')
plt.subplot(235)
plt.imshow(y_train_slice_bac, cmap='gray')
plt.subplot(236)
plt.imshow(y_train_slice_edge, cmap='gray')
plt.show()








