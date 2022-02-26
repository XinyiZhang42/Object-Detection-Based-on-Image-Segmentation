'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    ''' Instructions: No one said data collection was easy. This code needs to be run 35 times
        for 7 classes. Each time, change the for loop 3 times to go through all the data. Then
        change class to next one. Repeat for all 8 classes.'''

    # initialize list storage variables that are for all images
    #Change below class name as needed before running
    h_recycling_bin_blue_pixels = []
    s_recycling_bin_blue_pixels = []
    v_recycling_bin_blue_pixels = []

    #Change below loop before running - make sure to change save file name
    for i in range(1,21): #RUN FIRST TIME USING THIS
    #for i in range(21,41): # RUN SECOND TIME USING THIS
    #for i in range(41,61): # RUN THIRD TIME USING THIS
        # read training images
        folder = 'data/training'
        filename = '{:04d}'.format(i) + '.jpg'
        img = cv2.imread(os.path.join(folder, filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #CHANGE THIS LINE WHEN CHANGING COLORSPACES
        #print('img shape:', img.shape) #ixjx3
        #print('img:', img[3][3][:]) #uint8

        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

        # get the image mask
        mask = my_roi.get_mask(img_hsv)
        #print('mask shape:', mask.shape) #ixj

        # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img_hsv[mask, :].shape[0])

        ax1.imshow(img)
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)

        plt.show(block=True)

        # save mask-positive and mask-negative rgb pixels into lists
        num_img_rows = img_hsv.shape[0]
        num_img_cols = img_hsv.shape[1]
        for i in range(num_img_rows):
            for j in range(num_img_cols):
                #print('i,j:',i,j)
                if mask[i,j] == 1: #True
                    # Change below class name as needed before running
                    h_recycling_bin_blue_pixels.append(img_hsv[i,j,0])
                    s_recycling_bin_blue_pixels.append(img_hsv[i,j,1])
                    v_recycling_bin_blue_pixels.append(img_hsv[i,j,2])
                    # below is just to check outputs are correct
                    #print('h_recycling_bin_blue_pixels:', h_recycling_bin_blue_pixels)
                    #print('h_recycling_bin_blue_pixels type:', type(h_recycling_bin_blue_pixels))

    #convert lists to numpy arrays
    #Change below class name as needed before running
    h_recycling_bin_blue_pixels_np = np.asarray(h_recycling_bin_blue_pixels)
    s_recycling_bin_blue_pixels_np = np.asarray(s_recycling_bin_blue_pixels)
    v_recycling_bin_blue_pixels_np = np.asarray(v_recycling_bin_blue_pixels)

    #stack rgb numpy arrays vertically
    #Change below class name as needed before running
    recycling_bin_blue_pixels = np.vstack((h_recycling_bin_blue_pixels_np,s_recycling_bin_blue_pixels_np,v_recycling_bin_blue_pixels_np))

    #transpose numpy arrays from 3xk into kx3 dimensions, k = however many pixels are bin
    #Change below class name as needed before running
    recycling_bin_blue_pixels = recycling_bin_blue_pixels.transpose()

    # save pixel rgb data of the classes
    # Change below class name and path as needed before running
    np.save('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set1.npy', recycling_bin_blue_pixels) #RUN FIRST TIME USING THIS
    #np.save('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set2.npy', recycling_bin_blue_pixels) #RUN SECOND TIME USING THIS
    #np.save('./data/training/hsv_color_space_8_classes/recycling_bin_blue_pixels_hsv_set3.npy', recycling_bin_blue_pixels) #RUN THIRD TIME USING THIS