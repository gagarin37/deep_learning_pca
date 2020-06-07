#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 08:30:48 2019

@author: yuri tolkach
"""

# =============================================================================
# 1. SET PARAMETERS
# =============================================================================
p_s = 600
###MODEL PATCH SIZE
m_p_s = 350
###MODEL PATH (Type 2 model for Gleason grading)
model_dir = '' #to be defined
model_name = '' #to be defined
###DIRECTORY WITH IMAGES
base_dir = '' #to be defined
###OUTPUT DIRECTORY FOR RESULTS FILE
result_dir = '' #to be defined
###NAME OF THE OUTPUT FILE
result_name = "GS_model_name_btstrp"

###Bootstrapping parameters:
###Number of bootstrapping rounds
bstr_rounds = 20


# =============================================================================
# 2. IMPORT LIBRARIES
# =============================================================================
from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
from PIL import Image
import staintools
import random


# =============================================================================
# 3. LOAD MODEL
# =============================================================================
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
#present summary of the model
model.summary()


# =============================================================================
# 4. INITIALIZE STAIN NORMALIZER
# =============================================================================
#Standartization image
st = staintools.read_image('standard_he_stain_small.jpg')
#Inititate Brightness Standardizer
standardizer = staintools.BrightnessStandardizer()
#Inititate Stain Normalizer using "macenko" principle
stain_norm = staintools.StainNormalizer(method='macenko')
#Read HE staining schema from Standartization image
stain_norm.fit(st)


# =============================================================================
# 5. FUNCTIONS
# =============================================================================

#FUNCTION: Crop stem image (large tumor image)
#and return it with size in patches
#Necessary as some images sized do not perfectly pass into Sx600px frame
def image_crop (im):
    w, h = im.size    
    w_p = int(w/600)
    h_p = int(h/600)
    #Centroid of the image        
    w_center = int(w / 2)
    h_center = int(h / 2)
    #Coordinates of upper left corner of the cropped image
    coord_w_1 = int(w_center - (w_p / 2 * 600))
    coord_h_1 = int(h_center - (h_p / 2 * 600))
    #Area with ALL coordinates, based on coordinates of upper left corner
    area = (coord_w_1, coord_h_1 , coord_w_1 + w_p*600, coord_h_1 + h_p * 600)
    #Crop image and return it back
    im_crop = im.crop (area)
    return (im_crop, w_p, h_p)


#FUNCTION: Randomly generate list of coordinates of patches for bootstrapping
def coord_loop (coords, w_p, h_p, e, i):
    temp_1 = random.randint(1, w_p)
    temp_2 = random.randint(1, h_p)

    #Re-Generated the coordinated, if they already have been included
    #Do this until generated coordinates are unique
    while no_pass_fun (coords, e, i, temp_1, temp_2) == True:
        temp_1 = random.randint(1, w_p)
        temp_2 = random.randint(1, h_p)
        print("generating new coordinates")

    #save coordinates into coords array
    coords[i, e*2] = temp_1
    coords[i, e*2 + 1] = temp_2

#Function to check if the patch with such coordinates was already included
#into bootstrapping list
#If YES, than coordinates should be generated once more time and once more time tested
def no_pass_fun (coords, e, i, temp_1, temp_2):
    for x in range (e):
        if coords[i, x*2] == temp_1:
            if coords[i, x*2 + 1] == temp_2:
                return True
    return False
            

#Generate coords array which would be used to save coordinates for every
#bootstrapping round
#Size is dependent on current ROI size (1,2,3,n... patches) = bstr_size
#Fill it with randomly created coordinates
def bstr_coord_gen (bstr_rounds, bstr_size, w_p, h_p):
    coords = np.zeros((bstr_rounds, 2 * bstr_size), dtype=np.int16)
    for i in range(bstr_rounds):
        for e in range (bstr_size):
            coord_loop(coords, w_p, h_p, e, i)    
    return coords

#FUNCTION: prediction for single patches
def pred (model, patch):
    #PREPROCESSING
    patch = np.float32(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    #GS count
    counter_3 = round(preds [0,0] * 100, 1)
    counter_4 = round(preds [0,1] * 100, 1)
    counter_5 = round(preds [0,2] * 100, 1)
    score = (counter_3, counter_4, counter_5)
    return score


#FUNCTION: get pixel coordinates of single patches for boostrapping round
#"Patch" level coordinates were already randomly generated
#Now we need pixel coordinates to make extractions of patches from image
#for further test with the model
def get_px_coords (bstr_coords_l):
    area = np.zeros((bstr_size, 4), dtype = np.int16)
    for i in range(bstr_size):
        wb = bstr_coords_l [i*2]
        hb = bstr_coords_l [i*2 + 1]
        c1 = (wb - 1) * 600
        c2 = (hb - 1) * 600
        c3 = c1 + 600
        c4 = c2 + 600
        area [i] = (c1, c2, c3, c4)
    return area

#FUNCTION: gleason scoring from percentages of single patterns,
#returns Gleason Score ans WHO/ISUP-Grade group
def gscoring (score):
    
    gp3, gp4, gp5 = score
    
    if gp5 > 5: #Firstly, check if enough GP5 is present (>5%)
        if gp4 > 5: # Calculate if there is enough GP4 to be primary or secondary pattern
            if gp4 > gp5: # X+5, GS 3+5 is still possible
                if gp4 >= gp3: # 4+5
                    prim = 4
                    second = 5
                    ISUP = 5
                else: # 3+5
                    prim = 3
                    second = 5
                    ISUP = 4
            else: # 5+X (gp4 > 5, gp5 > 5, gp3 - not defined)
                if gp4 >= gp3: #5+4
                    prim = 5
                    second = 4
                    ISUP = 5
                else: 
                    prim = 5 #5+3, if gp4 > 5 and gp4 !>= gp3, then gp3 > 5
                    second = 3
                    ISUP = 4
        elif gp3 > 5: # In the absence of enough GP4; calculate if there is enough GP3 to be primary or secondary pattern
            if gp3 > gp5: # 3+5
                prim = 3
                second = 5
                ISUP = 4
            else: # 5+3
                prim = 5
                second = 3
                ISUP = 4
        else: #Both GP3 and GP4 are low (<5%)
            prim = 5 # 5+5
            second = 5
            ISUP = 5
 
    else: #GP5 is not present at >5%, therefore GS does not include GP5
        if gp3 > ((100-gp5) / 2): #GP3 is dominant
            prim = 3
            if gp4 > 5: #calculate secondary pattern in 3+X
                second = 4 # 3+4
                ISUP = 2
            else:
                second = 3
                ISUP = 1 # 3+3
        else: #GP4 is dominant
            prim = 4
            if (gp3 > 5): #calculate secondary pattern in 4+X
                second = 3 # 4+3
                ISUP = 3
            else:
                second = 4
                ISUP = 4 # 4+4
    
    #return (primary Gleason pattern, secondary Gleason pattern, ISUP_grade group)
    gs = (prim, second, ISUP)
    return gs



# =============================================================================
# 6. MAIN SCRIPT
# =============================================================================

#Retrieve file names of large tumor images
fnames = sorted(os.listdir(base_dir))

#Here 20 (19+1) is maximal size of subsampled ROI in patches (every patch 600x600px)
#consider that some images could have size less than defined maximal number of
#patches. This should be controled, otherwise you have an indefinite loop
#by generation of unique coordinates
#With this range test will include bootstrapping starting
#from minimal subsampled ROI size of 1 patch, then 2, 3, 4, ... and maximal
#size of 20 patches
for i in range(19):
    bstr_size = i + 1

    #Create output file path
    path_result_full = result_dir + result_name + "_" + str(bstr_size) + "_full.txt"
    path_result_short = result_dir + result_name + "_" + str(bstr_size) + "_short.txt"
    
    #Main script as Loop 
    for fname in fnames:
        
        filename = os.path.join(base_dir, fname)
        im = image.load_img(filename)
        print(fname, "loaded")
        
        #Crop large images to a size of whole patch number
        im, w_p, h_p  = image_crop (im)
        print ("Cropped to target size")    

        #Feedback
        print("Size in px:", im.size)
        print("Size:", w_p, "x", h_p, "patches")
        print("Overall:", w_p * h_p, "patches")

        #Control if tumor image has overall size less than 16 patches,
        #if YES then skip it.
        #16 patches was a minimal size of images in our dataset
        if w_p * h_p < 16:
            continue
        
        print("Starting bootstrapping with size:", bstr_size, "patch(es);", "Bootstrapping rounds n =", bstr_rounds)
        
        #Get coordinates of random patches
        bstr_coords = bstr_coord_gen(bstr_rounds, bstr_size, w_p, h_p)
        #every line in bstr_coords is one bootstrapping round
        #in every single line coordinates of n patches, where n - size of ROI in patches
            
        #prepare containers for short (only WHO/ISUP grade group) and full output
        output_0 = fname + "\t" + str(w_p * h_p) + "\t"
        output_full = ""
        output_short = ""
        
        for l in range(len(bstr_coords)): # length here is a number of rounds during bootstrapping
            #get coordinates of patches
            px_coords = get_px_coords (bstr_coords [l]) #px_coords is a numpy array
            #every line = coordinates of single patch = area (w1,h2,w2,h2)
            
            #Counter (percentage of Gleason patterns) is a global for every ROI within the Round of boostrapping
            #(=single line in bstr_coords)
            counter_3 = 0
            counter_4 = 0
            counter_5 = 0
            
            #Analyse single patches from one ROI of bootstrapping round and update global counter for ROI
            for j in range(len(px_coords)):
            
                #generate patch as image using pixel coordinates
                patch = im.crop (px_coords[j])        

                #normalization, preprocessing
                patch = patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)        
                patch = np.array(patch)
                patch = standardizer.transform(patch)
                patch = stain_norm.transform(patch)
                
                #run model and get probilities of patterns
                score = pred (model, patch)
                
                #Update GS counter
                counter_3 = counter_3 + score [0]
                counter_4 = counter_4 + score [1]
                counter_5 = counter_5 + score [2]
            

            #Calculate percentage of patterns from counters
            gp3 = counter_3 / (counter_3 + counter_4 + counter_5) * 100
            gp4 = counter_4 / (counter_3 + counter_4 + counter_5) * 100
            gp5 = counter_5 / (counter_3 + counter_4 + counter_5) * 100
        
            gp3 = round (gp3, 1)
            gp4 = round (gp4, 1)
            gp5 = round (gp5, 1)

            score = (gp3, gp4, gp5)
                   
            #transform score into Gleason Score (primary and secondary pattern)
            #and WHO/ISUP grade group
            gs_1, gs_2, isup = gscoring(score)
                
            #Create output, short (only ISUP) and full (Percentage patterns, GS, ISUP) versions
            output_full = output_full + str(gp3) + "\t" + str(gp4) + "\t" + str(gp5) + "\t" + str(gs_1) + "\t" + str(gs_2) + "\t" + str(isup) + "\t"
            output_short = output_short + str(isup) + "\t"
            
        #Write out output into the file
        output_end_full = output_0 + output_full + "\n"
        results = open (path_result_full, "a+")
        results.write(output_end_full)
        results.close()
        
        output_end_short = output_0 + output_short + "\n"
        results = open (path_result_short, "a+")
        results.write(output_end_short)
        results.close()