#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""

#0. SET PARAMETERS
###Path to directory with models (one or several models to be tested)
model_dir = ''
###DIRECTORY WITH IMAGES
#Tumor
base_dir_tu = ''
#Benign
base_dir_norm = ''
###OUTPUT DIRECTORY FOR RESULT FILES
result_dir = ''
###

#1. IMPORT LIBRARIES
from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
import staintools
from statistics import median


#2. GENERATE LISTS AND NAMES
###GENERATE LIST OF MODELS (if several models are tested)
model_names = sorted(os.listdir(model_dir))
###MODEL PATCH SIZES: define the patch size to use within models
#here for example two models in list, each working with 350px patches
m_p_s_list = [350, 350]

#3. INITIALIZE STAIN NORMALIZER
#Standartization image
st = staintools.read_image('standard_he_stain_small.jpg')
#Inititate Brightness Standardizer
standardizer = staintools.BrightnessStandardizer()
#Inititate StainNormalizer "macenko"
stain_norm = staintools.StainNormalizer(method='macenko')
#Read Hematoxylin/Eosin staining schema from Standartization image
stain_norm.fit(st)

#4. FUNCTIONS
#Implementation of the Strategy C8 (derivates of the main image, s. Methods)
#as a function
#As input: native version of the patch
def gateway_median (patch):
    #native version of the patch (base)
    base = patch #1
    #rotation derivates
    r90 = patch.rotate(90) #2
    r180 = patch.rotate(180) #3
    r270 = patch.rotate(270) #4
    #flip/rotation derivates
    r90_VF = ImageOps.flip(r90) #5
    r270_VF = ImageOps.flip(r270) #6
    #flip derivates
    VF = ImageOps.flip(base) #7
    HF = base.transpose(Image.FLIP_LEFT_RIGHT) #8 
    #calculate final predictions from predictions of individual derivates
    #pred() function refers to the generation of predictions for classes by model 1
    #Tumor vs benign tissue
    #Median to generate final predictions
    pred_stack = np.vstack((pred(base),pred(r90),pred(r180),pred(r270),pred(r90_VF),pred(r270_VF),pred(VF),pred(HF)))
    pred_1 = median(pred_stack[0:8,0])
    pred_2 = median(pred_stack[0:8,1])
    pred_3 = median(pred_stack[0:8,2])
    preds_med = np.array([pred_1, pred_2, pred_3])
    #returns final predictions
    return preds_med 

#Function for generation of prediction for single patches (used in C8)
def pred (patch):
    #IMAGE TO ARRAY, PREPROCESSING
    patch = image.img_to_array(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    #return predictions
    return preds

#Loop for analysis of patches from validation dataset with "tumor" label
#as function
def processor_tu (m_p_s):
    #define work directory
    work_dir = base_dir_tu
    #define output containers
    o_C1 = output_tu_C1
    o_C8 = output_tu_C8
    
    #read names from the directory
    fnames = sorted(os.listdir(work_dir))

    #Analysis loop
    for fname in fnames:
        filename = os.path.join(work_dir, fname)
        im = image.load_img(filename)
        #preprocessing of the image
        im = im.resize((m_p_s,m_p_s), Image.ANTIALIAS)
        im = np.array(im)
        #stain normalization
        im = standardizer.transform(im)
        im = stain_norm.transform(im)
        #for eventual further C8 analysis
        im_sn = Image.fromarray(im)
        
        im = np.float32(im)
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis = 0)
        x /= 255.
        
        #prediction
        preds = model.predict(x)
        #pred gland
        pr_1 = str(round(preds[0,0],3))
        #pred non-gland
        pr_2 = str(round(preds[0,1],3))
        #pred tumor
        pr_3 = str(round(preds[0,2],3))
        
        #Output of C1 (analysis of naative version of the patch)
        output = fname + "\t" + pr_1 + "\t" + pr_2 + "\t" + pr_3 + "\n"
        
        #Write down output of C1
        results = open (o_C1, "a+")
        results.write(output)
        results.close()
        
        #Additional analysis using C8 strategy (see Methods) if prediction
        #probability for tumor class in the gray zone 0.2-0.5
        if preds[0,2] < 0.5:
            #Enter C8 pipeline
            if preds[0,2] > 0.2:
                print(fname, " was misclassified, but trying C8") # feedback
                #start C8 analysis (function)
                preds_C8 = gateway_median(im_sn)
                
                #Write down results of C8 analysis.
                output_C8 = fname + "\t" + str(preds_C8[0]) + "\t" + str(preds_C8[1]) + "\t" + str(preds_C8[2]) + "\n"
                results = open (o_C8, "a+")
                results.write(output_C8)
                results.close()
            
                if preds_C8[2] > 0.5:
                    print(fname, " was reclassified") #feedback from analysis using C8 strategy
                else:
                    print(fname, " was not reclassified") #feedback from analysis using C8 strategy
            else:
                print(fname, " was misclassified") #feedback from analysis using C8 strategy
                
        else:
            print(fname, " is a tumor") #feedback
        

#Loop for analysis of patches from validation dataset with "benign" label
#as function
#Analogous to tumor processor above
def processor_norm (m_p_s):
    work_dir = base_dir_norm
    o_C1 = output_n_C1
    o_C8 = output_n_C8
    
    fnames = sorted(os.listdir(work_dir))
    
    for fname in fnames:
        filename = os.path.join(work_dir, fname)
        im = image.load_img(filename)
        im = im.resize((m_p_s,m_p_s), Image.ANTIALIAS)
        im = np.array(im)
        #stain normalization
        im = standardizer.transform(im)
        im = stain_norm.transform(im)
        #for C8
        im_sn = Image.fromarray(im)
        
        im = np.float32(im)
        x = image.img_to_array(im)
        x = np.expand_dims(x, axis = 0)
        x /= 255.
        
        #prediction
        preds = model.predict(x)
        #pred gland
        pr_1 = str(round(preds[0,0],3))
        #pred non-gland
        pr_2 = str(round(preds[0,1],3))
        #pred tumor
        pr_3 = str(round(preds[0,2],3))
        
        #Output of C1
        output = fname + "\t" + pr_1 + "\t" + pr_2 + "\t" + pr_3 + "\n"
        
        #Write down output of C1
        results = open (o_C1, "a+")
        results.write(output)
        results.close()
        
        if preds[0,2] > 0.5:
            #Enter C8 pipeline
            if preds[0,0] > 0.2 or preds[0,1] > 0.2:
                print(fname, " was misclassified, but trying C8")
                #start C8 program
                preds_C8 = gateway_median(im_sn)
                
                #Write down results of C8 independent of classification from function output.
                output_C8 = fname + "\t" + str(preds_C8[0]) + "\t" + str(preds_C8[1]) + "\t" + str(preds_C8[2]) + "\n"
                results = open (o_C8, "a+")
                results.write(output_C8)
                results.close()
            
                if preds_C8[2] < 0.5:
                    print(fname, " was reclassified")
                else:
                    print(fname, " was not reclassified")
            else:
                print(fname, " was misclassified")    
                
        else:
            print(fname, " is normal")
        

#MAIN LOOP
i = 0
for model_name in model_names:
    #1. Load model from list (one or more)
    print("Loading model: ", model_name, " ...")
    path_model = os.path.join(model_dir, model_name)
    model = load_model(path_model)
   
    #Create path to results file
    output_tu_C1 = result_dir + model_name + "__C1__tu.txt"
    output_n_C1 = result_dir + model_name + "__C1__norm.txt"
    output_tu_C8 = result_dir + model_name + "__C8__tu.txt"
    output_n_C8 = result_dir + model_name + "__C8__norm.txt"

    #start analysis
    processor_tu(m_p_s_list[i])
    processor_norm(m_p_s_list[i])
    
    #Increment of i for m_p_s
    i = i+1
    print("Ready! Going to the next model.") #feedback

    
    
    
