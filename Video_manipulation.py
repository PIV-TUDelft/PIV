# -*- coding: utf-8 -*-

# Gamma correction, a translation between the sensitivity of our eyes and sensors of a camera.
# Linear to non-linear relationship
# Gamma values < 1 will shift the image towards the darker end of the spectrum. 
# Gamma values > 1 will make the image appear lighter. 
# A gamma value of G=1 will have no affect on the input image.

import numpy as np
import cv2
import os

# Gamma function:
def adjust_gamma(image, gamma=1.5):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

#------------------------------------------------------------------------------

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('Dommel_KH.MP4');

#to stop duplicates
currentFrame = 0

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

File_Output = 'output_Dommel_KH.MP4'
out = cv2.VideoWriter(File_Output, -1, 24, (frame_width, frame_height))

# Check if camera opened successfully
while(cap.isOpened()):    
    # capture frame by frame
    ret, frame = cap.read() #if the frame is available ret will be true
    if ret == True:
        
        #gamma correction
        gamma = adjust_gamma(frame, gamma = 0.15) 
    
        #grayscale
        gray = cv2.cvtColor(gamma, cv2.COLOR_RGB2GRAY)
    
        #Write the frame into the file 'output.avi'
        out.write(gray)
        
        #showing all images as video
        cv2.imshow('frame', gray) 
        
        # Saves image of the current frame in jpg file
        name = 'C:/Users/User/Documents/Universiteit STUDIE/TU Delft/MSC Thesis/PythonPIV/Dommel_Images_KH/' + 'Dommel_KH' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, gray)
        
        #wait a short moment after each image
        #& quit the video with Q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        # to stop duplicate images
        currentFrame += 1

#release everyting if job is finished
cap.release()
out.release()
cv2.destroyAllWindows() 

#------------------------------------------------------------------------------
