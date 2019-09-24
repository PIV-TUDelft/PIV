# -*- coding: utf-8 -*-
import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab

from tqdm import tqdm # the progress bar
from openpiv import tools, process, scaling, pyprocess, validation, filters

#------------------------------------------------------------------------------

def remove_outliers(u, v, mask):
    """
    Removal of outliers. Points are indicated as outlier if u or v
    is off two times the standard deviation from the median.
    
    This function is placed after validation.sig2noise_val()
    
    Input:
    ------
    u, v, mask prefereably as output of validation.sig2noise_val()
    
    Output:
    -------
    u, v, mask without outliers
    """

    u_nonan = u.copy()
    v_nonan = v.copy()

    u_no_outliers = u.copy()
    v_no_outliers = v.copy()
    mask_no_outliers = mask.copy()

    u_nonan = u_nonan[~np.isnan(u_nonan)]
    v_nonan = v_nonan[~np.isnan(v_nonan)]

    u_median = np.median(u_nonan)
    u_std = np.std(u_nonan)

    v_median = np.median(v_nonan)
    v_std = np.std(v_nonan)

    for i in range(len(u)):
        for j in range(len(u[i])):
            if u[i,j] > u_median + 2*u_std or u[i,j] < u_median - 2*u_std:
                mask_no_outliers[i,j] = True
                u_no_outliers[i,j] = np.nan
                v_no_outliers[i,j] = np.nan

            if v[i,j] > v_median + 2*v_std or v[i,j] < v_median - 2*v_std:
                mask_no_outliers[i,j] = True
                u_no_outliers[i,j] = np.nan
                v_no_outliers[i,j] = np.nan

    return u_no_outliers, v_no_outliers, mask_no_outliers

#------------------------------------------------------------------------------

# Import all images from this file
# Returns a list containing the names of the entries in the directory given
path_data= 'C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH'
images = os.listdir(path_data)

# set-up parameters
winsize        = 40             # Pixels, size of the interrogation window in image A
searchsize     = winsize        # Pixels, size of the search area in image B
overlap        = 10             # Pixels, overlap between adjacent windows
frame_rate     = 24             # Frequence of the video(frames per second)
dt             = 1./frame_rate  # Time interval

#1080p = 1980 x 1080 pixels
#4K    = 3840 x 2160 pixels
scaling_factor = 334           # Pixels per metre

#------------------------------------------------------------------------------

# Function that processes the images
def PIV(image_0, image_1, winsize, searchsize, overlap, frame_rate, scaling_factor):
     
    frame_0 = image_0
#     [0:600, :]
    frame_1 = image_1
#     [0:600, :]
    
    # Processing the images with interrogation area and search area / cross correlation algortihm
    u, v, sig2noise = pyprocess.extended_search_area_piv(frame_0,
                                                         frame_1,
                                                         window_size=winsize,
                                                         overlap=overlap,
                                                         dt=dt,
                                                         search_area_size=searchsize,
                                                         sig2noise_method='peak2peak')
    
    # Compute the coordinates of the centers of the interrogation windows
    x, y = pyprocess.get_coordinates(image_size=frame_0.shape,
                                     window_size=winsize,
                                     overlap=overlap)
    
    # This function actually sets to NaN all those vector for 
    # which the signal to noise ratio is below 1.3. 
    u, v, mask = validation.sig2noise_val(u,
                                          v,
                                          sig2noise,
                                          threshold=1.2)
    
    # Function as described above, removing outliers deviating with more 
    # than twice the standard deviation
    u, v, mask = remove_outliers(u, v, mask) 
    
    # Replacing the outliers with interpolation 
    u, v = filters.replace_outliers(u,
                                    v,
                                    method='localmean',
                                    max_iter=50,
                                    kernel_size=3)
                                   
    # Apply an uniform scaling to the flow field to get dimensional units
    x, y, u, v = scaling.uniform(x,
                                 y,
                                 u,
                                 v,
                                 scaling_factor=scaling_factor)
    
    return x, y, u, v, mask

#------------------------------------------------------------------------------

# with tqdm(total=i_max) as pbar:
# N = 2
N = len(images)-1

u_sum = 0
v_sum = 0

# Reading the images from the file and run them through the PIV function
# Save the vectors in a .txt file
with tqdm(total=N) as pbar: # use the progress bar
    for n in range(N): # loop through all images
        image_0 = tools.imread('C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH/'+str(images[n]))
        image_1 = tools.imread('C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH/'+str(images[n+1]))

        x, y, u, v, mask = PIV(image_0, image_1, winsize, searchsize, overlap, frame_rate, scaling_factor)

        tools.save(x,
                   y,
                   u,
                   v,
                   mask,
                   'C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Results_Dommel/Dommel_KH' +str(n)+ '.txt')

        # sum the vector values of all images 
        u_sum += u
        v_sum += v
        
        pbar.update(1)
        
# Determine averages
u_avg = u_sum/N
v_avg = v_sum/N
mask_avg = np.zeros_like(mask)

# save the average vectors in a .txt file
tools.save(x*winsize,
           y*winsize,
           u_avg,
           v_avg,
           mask_avg,
           'C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH/average.txt')

# display the vector field that is saved (blue field)
tools.display_vector_field('C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH/average.txt',
#                            on_img=True,
#                            image_name='Brenta/'+str(images[0]),
#                            window_size=winsize,
#                            scaling_factor=scaling_factor,
                           scale=6,
                           width=0.0035,
                          )
plt.savefig('average.jpg')

plt.axis('scaled')

plt.plot(u, v,'ko', ms=1)
plt.axis('scaled')
plt.xlabel('u')
plt.ylabel('v')
plt.savefig('deviation_U_V')

fig=plt.figure(figsize=(9,4))
# norm = matplotlib.colors.Normalize(vmin=0,vmax=1.,clip=False)
color=np.sqrt(v_avg**2 + u_avg**2)

plt.quiver(x, y, u_avg, v_avg, color)
plt.axis('scaled')
plt.colorbar();

plt.savefig('Vector_field.jpg')