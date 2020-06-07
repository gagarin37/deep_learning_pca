
###BACKGROUND / TISSUE and CELL DETECTOR FUNCTIONS

import numpy as np

#1.Tissue detector
#analyses distribution of pixels colours in the patch
#if tissue staining patterns are detected above threshold
#than patch classified as "tissue", otherwise "background"
#threshold tuned to detect even small fragments of tissue

def tissue_detector (im):
    count_med = 0
    im_B = im[:,:,2]
    im_B_reshape = np.reshape(im_B, len(im_B[0])*len(im_B[1]))
    for i in range(len(im_B_reshape)):
        if im_B_reshape[i] < 200 and im_B_reshape[i] > 100:
            count_med = count_med + 1
            if count_med > 7000:
                return True
    return False

#2. Nucleus detector (Blue/Hematoxylin colours Detector)
#necessary for decision of brightness / stain normalization
#if number of pixels corresponding to nuclear staining patterns of the cells
#then practically there are nuclei detected
#is above the threshold, than patch contains cells

def blue_detector (im):
    count = 0
    im_B = im[:,:,2]
    im_B_reshape = np.reshape(im_B, len(im_B[0])*len(im_B[1]))
    for i in range(len(im_B_reshape)):
        if im_B_reshape[i] < 100:
            count = count + 1
            if count > 10:
                return True
    return False