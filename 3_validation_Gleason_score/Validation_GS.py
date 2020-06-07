#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: yuri tolkach

#0. SET PARAMETERS
#######################
###PATCH SIZE
p_s = 600
###MODEL PATCH SIZE
m_p_s = 350
###MODEL_PATH
model_dir = '' #should be defined
model_name = '' #should be defined
###DIRECTORY WITH IMAGES
base_dir = '' #should be defined
###OUTPUT DIRECTORY FOR RESULTS FILE
result_dir = '' #should be defined
###NAME OF THE OUTPUT FILE
result_name = '' #should be defined

#1. IMPORT LIBRARIES
from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
from PIL import Image
import staintools

#2. LOAD_MODEL
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
#display model summary
model.summary()

#########################
#3. STAIN NORMALIZER
#Standartization image
st = staintools.read_image('standard_he_stain_small.jpg')
#Initialize Brightness Standardizer
standardizer = staintools.BrightnessStandardizer()
#Initialize Stain Normalizer "macenko"
stain_norm = staintools.StainNormalizer(method='macenko')
#Read staining schema from Standartization image
stain_norm.fit(st)
#########################


#4. MAIN SCRIPT AS A LOOP

#Get file names in working directory
fnames = sorted(os.listdir(base_dir))

#Create output file path (base directory), the last symbol in the file name is changing
path_result = result_dir + result_name + ".txt"

#Main script as Loop
for fname in fnames:
    print(fname)
    filename = os.path.join(base_dir, fname)
    im = image.load_img(filename)

    #retrieve image size
    w_img, h_img = im.size

    #calculate number of patches-to-analyze using image dimensions
    patch_n_w = int(w_img / p_s)
    patch_n_h = int(h_img / p_s)

    #Counters of gleason patterns for the whole image defined (s. Algorithm
    #description in Methods)
    counter_3 = 0
    counter_4 = 0
    counter_5 = 0

    #Analysis loop
    #Systematic analysis of all patches in the image
    for hi in range(patch_n_h):
        h = hi*p_s + 1
        if (hi==0):
            h = 0

        #feedback
        print("Current cycle ", hi+1, " of ", patch_n_h)

        for wi in range(patch_n_w):
            w = wi*p_s+1
            if (wi==0):
                w = 0

            #Extract patch for analysis
            #Coordinates of the patch: left, up, right, bottom
            area = (w,h,w+p_s,h+p_s)
            work_patch = im.crop(area)

            #Resize to model patch size (depends on target magnification)
            work_patch = work_patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)

            #Preprocessing
            wp_temp = np.array(work_patch)
            #Brightness and stain normalization
            wp_temp = standardizer.transform(wp_temp)
            wp_temp = stain_norm.transform(wp_temp)
            wp_temp = np.float32(wp_temp)

            #IMAGE TO ARRAY, PREPROCESSING
            wp_temp = np.expand_dims(wp_temp, axis = 0)
            wp_temp /= 255.

            #Get predictions from model (predictions for pattern 3, 4 and 5)
            preds = model.predict(wp_temp)

            #GS count (counting the whole image score according to Algorithm)
            counter_3 = counter_3 + (preds [0,0] * 100)
            counter_4 = counter_4 + (preds [0,1] * 100)
            counter_5 = counter_5 + (preds [0,2] * 100)


    #Calculate percentages of Gleason patterns for the whole image
    num_patches = patch_n_w * patch_n_h

    perc_GP_3 = (counter_3 / (counter_3 + counter_4 + counter_5) * 100)
    perc_GP_4 = (counter_4 / (counter_3 + counter_4 + counter_5) * 100)
    perc_GP_5 = (counter_5 / (counter_3 + counter_4 + counter_5) * 100)

    perc_GP_3 = round (perc_GP_3, 2)
    perc_GP_4 = round (perc_GP_4, 2)
    perc_GP_5 = round (perc_GP_5, 2)

    counter_3 = round (counter_3, 2)
    counter_4 = round (counter_4, 2)
    counter_5 = round (counter_5, 2)

    #feedback during script implementation
    print ("Number of patches: ", num_patches)
    print ("GP3: ", perc_GP_3)
    print ("GP4: ", perc_GP_4)
    print ("GP5: ", perc_GP_5)
    print ("Counter 3: ", counter_3)
    print ("Counter 4: ", counter_4)
    print ("Counter 5: ", counter_5)

    #Save output into file in the tab-delimited format
    output = fname + "\t" + str(num_patches) + "\t" + str(perc_GP_3) + "\t" + str(perc_GP_4) + "\t" + str(perc_GP_5) + "\t" + str(counter_3) + "\t" + str(counter_4) + "\t" + str(counter_5) + "\n"
    results = open (path_result, "a+")
    results.write(output)
    results.close()