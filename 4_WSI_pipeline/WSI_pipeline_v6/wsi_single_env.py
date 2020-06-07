#IMPLEMENTATION OF ADDITIONAL STRATEGY TO REDUCE FALSE-POSITIVE RESULTS
#SINGLE POSITIVE PATCH ENVIRONMENT ANALYSIS STRATEGY (s. Methods)

#IMPORT LIBRARIES
import numpy as np
from keras_preprocessing import image
from PIL import Image

#FUNCTION: analysis of environment of single tumor patches to generate new binary map
def check_single_environment (model, slide, wsi_map_bin_f, p_s, m_p_s):
    #Write down coordinates of single tumor patches
    for he in range(wsi_map_bin_f.shape[0]):
        for wi in range(wsi_map_bin_f.shape[1]):
            if wsi_map_bin_f [he,wi] == 9999:   # = tumor
                #test environment function: to detect if neighbouring patches are also tumor
                if test_env (wsi_map_bin_f, he, wi) == False: #False = no further tumor patches in environment
                    #Starting evaluation of patch environment:
                    #4 newly generated neighbouring patches including parts of the patch-in-question
                    if neighb_check (model, slide, wi, he, p_s, m_p_s) == False: # False = no tumor in evironment
                        wsi_map_bin_f [he, wi] = 1111 #reclassify as benign
    return wsi_map_bin_f  #return updated map
                    


#FUNCTION: Test if there are further tumor patches in environment.
def test_env (wsi_map_bin_f, he, wi):
    #counter of positive neigbours
    counter = 0
    #define "patch" coordinates of 4 neighbour patches
    he_list = [he - 1, he, he + 1, he]
    wi_list = [wi, wi + 1, wi, wi - 1]
    #Get status of neigbours (tumor or benign)
    for i in range(4):
        if is_tumor(wsi_map_bin_f, he_list[i], wi_list[i]) == True:
            counter = counter + 1
    #Return True is environment is positive for tumor patches (1 or more are tumor)
    if counter > 0:
        return True
    else:
        return False

#FUNCTION support for test_env(): is patch a tumor?
def is_tumor (wsi_map_bin_f, he, wi):
    if wsi_map_bin_f [he, wi] == 9999:
        return True
    else:
        return False
    

#MAIN FUNCTION for test of newly generated patches from environment of the target patch
#S. Manuscript Methods and Figures for principle
def neighb_check (model, slide, wi, he, p_s, m_p_s):
    #get pixel coordinates of target patch
    w_c, h_c = wi * 600 + 300, he * 600 + 300
    #get pixel coordinates of patches from environment of the target patch
    #and generate them via read_region function of openslide package
    neighb_1 = slide.read_region((w_c - 600, h_c - 600), 0, (p_s,p_s)).convert('RGB')
    neighb_2 = slide.read_region((w_c, h_c - 600), 0, (p_s,p_s)).convert('RGB')
    neighb_3 = slide.read_region((w_c, h_c), 0, (p_s,p_s)).convert('RGB')
    neighb_4 = slide.read_region((w_c - 600, h_c), 0, (p_s,p_s)).convert('RGB')
    #make predictions for newly generated patches from environment
    pred_stack = np.vstack((pred(model, neighb_1, m_p_s),
                            pred(model, neighb_2, m_p_s),
                            pred(model, neighb_3, m_p_s),
                            pred(model, neighb_4, m_p_s)))
    max_pred_tu = max(pred_stack[0:4,2])
    #if any of these patches is classified as tumor?
    if max_pred_tu >= 0.5:
        return True # = tumor in environment. No need to reclassify target patch.
    else:
        return False

    
#FUNCTION for model prediction for single patches
def pred (model, patch, m_p_s):
    patch = patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)
    #IMAGE TO ARRAY, PREPROCESSING
    patch = image.img_to_array(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    return preds                #return prediction probabilities
