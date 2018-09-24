# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:39:03 2018

@author: Jintram
"""

def build_data(bac_image_path, x_folder, y_folder):
    # %%
    # N_PIXELS_TO_SELECT = 1000
    DATA_SELECTION_STRIDE = 30
    
    import os
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread
    import sklearn.preprocessing as pp
    
    from tqdm import tqdm
    
    bac_image_path = 'C:/Users/Jintram/Temporary/ecoli_set1/'
    bac_image_path_x = bac_image_path + x_folder
    bac_image_path_y = bac_image_path + y_folder
    
    bac_imgs = os.listdir(bac_image_path_x)
        # note that I use the fact x and y images have the same name
    
    print("Creating training set..")
    
    # initialize
    sample_count=0
    data_x = [] #np.empty([len(bac_imgs)*3*N_PIXELS_TO_SELECT,21,21])
    #data_x[:] = np.nan
    data_y = [] #np.empty([len(bac_imgs)*3*N_PIXELS_TO_SELECT,1])
    # data_y[:] = np.nan
    # Loop over images
    with tqdm(total=len(bac_imgs)) as bar:
        for img_idx in range(len(bac_imgs)):
        
            # %% just show one image
            
            bacimg_x = np.array(imread(bac_image_path_x + bac_imgs[img_idx]))
            bacimg_y = np.array(imread(bac_image_path_y + bac_imgs[img_idx]))
            
            # feature scaling
            bacimg_x = np.reshape(pp.scale( bacimg_x.flatten(), axis=0, with_mean=True, with_std=True, copy=True ),
                                  np.shape(bacimg_x))
            
            if sample_count==1:
                plt.figure()
                plt.subplot(121)
                plt.imshow(bacimg_x, cmap='gray')
                plt.subplot(122)
                plt.imshow(bacimg_y, cmap='gray')
                plt.show()
            
            # %% Now create one test situation
            
            # Note that we want each category to be represented well
            
            # Perhaps first crop the image a bit to not get too many uniform background pixels
            # Note: indexing = row, column
            maxrow, maxcol = np.shape(bacimg_y) # size img
            idx_nonzero = bacimg_y.nonzero() # location colony
            # get sides + margin
            MARGIN=30
            crop_row1   = np.max([np.min(idx_nonzero[0])-MARGIN,0]) 
            crop_row2   = np.min([np.max(idx_nonzero[0])+MARGIN,maxrow])
            crop_col1   = np.max([np.min(idx_nonzero[1])-MARGIN,0])
            crop_col2   = np.min([np.max(idx_nonzero[1])+MARGIN,maxcol])
            
            bacimg_x_2 = bacimg_x[ crop_row1:crop_row2, crop_col1:crop_col2]
            bacimg_y_2 = bacimg_y[ crop_row1:crop_row2, crop_col1:crop_col2]
            
            if sample_count==1:
                plt.figure()
                plt.subplot(121)
                plt.imshow(bacimg_x_2, cmap='gray')
                plt.subplot(122)
                plt.imshow(bacimg_y_2, cmap='gray')
                plt.show()
                
            # now find the locations of the three categories (excl. margin)
            # ===
            maxrow_crop, maxcol_crop = np.shape(bacimg_y_2)
            MARGIN_MASK=10 # we cannot select pixels to close to boundary
            # obtain the indices
            idxs_bg   = np.nonzero(bacimg_y_2==0)
            idxs_bac  = np.nonzero(bacimg_y_2==1)
            idxs_edge = np.nonzero(bacimg_y_2==2)
            # sort them into rows and columns
            idxs_bg_i = idxs_bg[0]
            idxs_bg_j = idxs_bg[1]
            idxs_bac_i = idxs_bac[0]
            idxs_bac_j = idxs_bac[1]
            idxs_edge_i = idxs_edge[0]
            idxs_edge_j = idxs_edge[1]
            
            # apply margins (we should define a function to make this more pleasant)
            # ===
            sel_idx_i=np.logical_and(idxs_bg_i>MARGIN_MASK,idxs_bg_i<(maxrow_crop-MARGIN_MASK))
            sel_idx_j=np.logical_and(idxs_bg_j>MARGIN_MASK,idxs_bg_j<(maxcol_crop-MARGIN_MASK))
            sel_idx = np.logical_and(sel_idx_i,sel_idx_j)
            idxs_bg_i_sel = idxs_bg_i[sel_idx]
            idxs_bg_j_sel = idxs_bg_j[sel_idx]       
            
            sel_idx_i=np.logical_and(idxs_bac_i>MARGIN_MASK,idxs_bac_i<(maxrow_crop-MARGIN_MASK))
            sel_idx_j=np.logical_and(idxs_bac_j>MARGIN_MASK,idxs_bac_j<(maxcol_crop-MARGIN_MASK))
            sel_idx = np.logical_and(sel_idx_i,sel_idx_j)
            idxs_bac_i_sel = idxs_bac_i[sel_idx]
            idxs_bac_j_sel = idxs_bac_j[sel_idx] 
            
            sel_idx_i=np.logical_and(idxs_edge_i>MARGIN_MASK,idxs_edge_i<(maxrow_crop-MARGIN_MASK))
            sel_idx_j=np.logical_and(idxs_edge_j>MARGIN_MASK,idxs_edge_j<(maxcol_crop-MARGIN_MASK))
            sel_idx = np.logical_and(sel_idx_i,sel_idx_j)
            idxs_edge_i_sel = idxs_edge_i[sel_idx]
            idxs_edge_j_sel = idxs_edge_j[sel_idx] 
            
            # create training data per pixel (but select a subset)
            # ===
            # determine subset
            #n_pixels_to_select_2 = min(N_PIXELS_TO_SELECT,
            #                         len(idxs_bg_i_sel),len(idxs_bac_i_sel),len(idxs_edge_i_sel))    
            subset_idxs = list(range(1,len(idxs_bg_i_sel),DATA_SELECTION_STRIDE))
            #subset_idxs = list(range(1,len(idxs_bg_i_sel),len(idxs_bg_i_sel)//n_pixels_to_select_2))
            idxs_bg_i_sel_subset = idxs_bg_i_sel[subset_idxs]
            idxs_bg_j_sel_subset = idxs_bg_j_sel[subset_idxs]
            subset_idxs = list(range(1,len(idxs_bac_i_sel),DATA_SELECTION_STRIDE))
            #subset_idxs = list(range(1,len(idxs_bac_i_sel),len(idxs_bac_i_sel)//n_pixels_to_select_2))
            idxs_bac_i_sel_subset = idxs_bac_i_sel[subset_idxs]
            idxs_bac_j_sel_subset = idxs_bac_j_sel[subset_idxs]
            subset_idxs = list(range(1,len(idxs_edge_i_sel),DATA_SELECTION_STRIDE))
            #subset_idxs = list(range(1,len(idxs_edge_i_sel),len(idxs_edge_i_sel)//n_pixels_to_select_2))
            idxs_edge_i_sel_subset = idxs_edge_i_sel[subset_idxs]
            idxs_edge_j_sel_subset = idxs_edge_j_sel[subset_idxs]
                # note: note sure book-keeping is OK here, but rounding down makes sure there are
                # enough items
            
            # print(str(n_pixels_to_select_2))
            
            # %% loop over (selected subset of) pixels
            N_px = min(len(idxs_bg_i_sel_subset),
                       len(idxs_bac_i_sel_subset),
                       len(idxs_edge_i_sel_subset))
            for pix_idx in range(N_px):
            
                
                #  now select one of each + margins
                x_train_slice_bg = bacimg_x_2[(idxs_bg_i_sel_subset[pix_idx]-MARGIN_MASK):(idxs_bg_i_sel_subset[pix_idx]+MARGIN_MASK+1),
                                                (idxs_bg_j_sel_subset[pix_idx]-MARGIN_MASK):(idxs_bg_j_sel_subset[pix_idx]+MARGIN_MASK+1)]
                y_train_slice_bg = bacimg_y_2[idxs_bg_i_sel_subset[pix_idx],idxs_bg_j_sel_subset[pix_idx]]
                
                x_train_slice_bac = bacimg_x_2[(idxs_bac_i_sel_subset[pix_idx]-MARGIN_MASK):(idxs_bac_i_sel_subset[pix_idx]+MARGIN_MASK+1),
                                                (idxs_bac_j_sel_subset[pix_idx]-MARGIN_MASK):(idxs_bac_j_sel_subset[pix_idx]+MARGIN_MASK+1)]
                y_train_slice_bac = bacimg_y_2[idxs_bac_i_sel_subset[pix_idx],idxs_bac_j_sel_subset[pix_idx]]
                    
                x_train_slice_edge = bacimg_x_2[(idxs_edge_i_sel_subset[pix_idx]-MARGIN_MASK):(idxs_edge_i_sel_subset[pix_idx]+MARGIN_MASK+1),
                                                (idxs_edge_j_sel_subset[pix_idx]-MARGIN_MASK):(idxs_edge_j_sel_subset[pix_idx]+MARGIN_MASK+1)]
                y_train_slice_edge = bacimg_y_2[idxs_edge_i_sel_subset[pix_idx],idxs_edge_j_sel_subset[pix_idx]]
                
            
                #  now show what we're doing        
                if sample_count==1:
                    plt.figure()
                    plt.subplot(131)
                    plt.title('background, \ny_val='+str(y_train_slice_bg))
                    plt.imshow(x_train_slice_bg, cmap='gray')
                    plt.plot(10,10,'or',markersize=15, markeredgewidth=2,markerfacecolor="None")
                    plt.subplot(132)
                    plt.title('bacterium, \ny_val='+str(y_train_slice_bac))
                    plt.imshow(x_train_slice_bac, cmap='gray')
                    plt.plot(10,10,'or',markersize=15, markeredgewidth=2,markerfacecolor="None")
                    plt.subplot(133)
                    plt.title('edge, \ny_val='+str(y_train_slice_edge))
                    plt.imshow(x_train_slice_edge, cmap='gray')
                    plt.plot(10,10,'or',markersize=15, markeredgewidth=2,markerfacecolor="None")
                    
                # now create a data entry to save later
                # background
                data_x.append(x_train_slice_bg)
                data_y.append(y_train_slice_bg)
                #data_x[sample_count] = x_train_slice_bg
                #data_y[sample_count] = y_train_slice_bg
                sample_count=sample_count+1            
                # bacterial body
                data_x.append(x_train_slice_bac)
                data_y.append(y_train_slice_bac)
                #data_x[sample_count] = x_train_slice_bac
                #data_y[sample_count] = y_train_slice_bac
                sample_count=sample_count+1
                # bacterial edge
                data_x.append(x_train_slice_edge)
                data_y.append(y_train_slice_edge)
                #data_x[sample_count] = x_train_slice_edge
                #data_y[sample_count] = y_train_slice_edge        
                sample_count=sample_count+1
            
            bar.update()
    
    # select data
    #data_x = data_x[range(sample_count)]
    #data_y = data_y[range(sample_count)]
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    print("All done.")
    return data_x, data_y


"""
# stuff to run always here such as class/def
def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
"""
