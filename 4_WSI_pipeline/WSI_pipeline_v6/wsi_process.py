#MAIN LOOP TO PROCESS WSI
#Processes:
#-tiling
#-background / tissue detection
#-brightness/stain normalization
#-predictions for individual patches
#-make mathematical map with predictions (numpy array)
#-implement C8 strategy if necessary


#IMPORT LIBRARIES
import numpy as np
import wsi_detectors as det
import staintools
from wsi_stain_norm import standardizer, stain_norm
from PIL import Image
from wsi_maps import record_map_preds
from wsi_c8_functions import gateway_median


# =============================================================================
# C1 VERSION (only native version of the patch analysed by model (s. Methods)
# =============================================================================
def slide_process (model, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s):
    
    #INITIALIZE STAIN NORMALIZER
    st = staintools.read_image('images/standard_he_stain_small.jpg')
    stain_norm.fit(st)    

    #CREATE CHUNK FOR MAP WITH PREDICTIONS according to whole image size
    wsi_map_preds = np.zeros((patch_n_h_l0, patch_n_w_l0, 3) , dtype=np.float32)

    
    #Start loop with tiling of the whole image according to patch_size (p_s)
    #p_s = 600 px at 40x magnificaton
    #further reduction to working model patch size (m_p_s)

    for hi in range(patch_n_h_l0):
        h = hi*p_s + 1
        if (hi==0):
            h = 0
        print("Current cycle ", hi+1, " of ", patch_n_h_l0)
        for wi in range(patch_n_w_l0):
            w = wi*p_s+1
            if (wi==0):
                w = 0
    
            #Generate single patch, prepare for analysis
            work_patch = slide.read_region((w,h), 0, (p_s,p_s))
            work_patch = work_patch.convert('RGB')
    
            #Resize to model patch size (depends on target magnification)
            work_patch = work_patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)
    
            #Patch image to array
            wp_temp = np.array(work_patch)
    
            #Control: 1. Is image black? (e.g., background of Mirax images is typically black)
            #Control: 2. Tissue detector.
            if (det.tissue_detector (wp_temp) == True): #tissue is present (non-background)
                if (det.blue_detector (wp_temp) == True): #cells are present
                    wp_temp = standardizer.transform(wp_temp)
                    wp_temp = stain_norm.transform(wp_temp)
    
                    wp_temp = np.float32(wp_temp)
    
                    #PREPROCESSING
                    wp_temp = np.expand_dims(wp_temp, axis = 0)
                    wp_temp /= 255.
    
                    #prediction from model (Type 1)
                    preds = model.predict(wp_temp)
                    #three classe: tumor, benign glandular, benign non-glandular

                    #record predictions into map
                    record_map_preds(wsi_map_preds, hi, wi, preds)

    return (wsi_map_preds)

# =============================================================================
# C8 version (version with implementation of C8 strategy,s Methods)
# =============================================================================

def slide_process_C8 (model, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s):
    
    #INITIALIZE STAIN NORMALIZER
    st = staintools.read_image('images/standard_he_stain_small.jpg')
    stain_norm.fit(st)    

    #CREATE CHUNK FOR MAP WITH PREDICTIONS
    wsi_map_preds = np.zeros((patch_n_h_l0, patch_n_w_l0, 3) , dtype=np.float32)

    
    #Start loop
    for hi in range(patch_n_h_l0):
        h = hi*p_s + 1
        if (hi==0):
            h = 0
        print("Current cycle ", hi+1, " of ", patch_n_h_l0)
        for wi in range(patch_n_w_l0):
            w = wi*p_s+1
            if (wi==0):
                w = 0
    
            #Generate patch
            work_patch = slide.read_region((w,h), 0, (p_s,p_s))
            work_patch = work_patch.convert('RGB')
    
            #Resize to model patch size (depends on target magnification)
            work_patch = work_patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)
    
            #Patch image to array
            wp_temp = np.array(work_patch)
    
            #Control: 1. Is image black? (background of Mirax images is typical black)
            #Control: 2. Tissue detector.
            if (det.tissue_detector (wp_temp) == True):
                #stain normalization
                if (det.blue_detector (wp_temp) == True):
                    wp_temp = standardizer.transform(wp_temp)
                    wp_temp = stain_norm.transform(wp_temp)
                    
                    im_sn = Image.fromarray(wp_temp)
                    
                    wp_temp = np.float32(wp_temp)
    
                    #PREPROCESSING
                    wp_temp = np.expand_dims(wp_temp, axis = 0)
                    wp_temp /= 255.
    
                    #prediction from model
                    preds = model.predict(wp_temp)

                    #record predictions into map using function
                    record_map_preds(wsi_map_preds, hi, wi, preds)

                    #if patch in gray zone
                    #(Tumor Class probability is between 0.2 and 0.8)
                    #Analyse through C8 algorithm and update predictions
                    if (preds [0,2]) >= 0.2 and (preds [0,2]) < 0.8:
                        preds_C8 = gateway_median(model, im_sn)
                        #update predictions after C8 analysis
                        record_map_preds(wsi_map_preds, hi, wi, preds_C8)

    return (wsi_map_preds) #returns mathematical map with predictions