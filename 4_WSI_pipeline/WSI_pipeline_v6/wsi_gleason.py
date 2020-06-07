from PIL import Image
import numpy as np
from keras_preprocessing import image
import staintools
from wsi_stain_norm import standardizer, stain_norm

#Input map = mathematical binary map of the slide,
#where 0 = background, 1111 = benign, 9999 = tumor
#other parametersm such as model, slide, and patch sizes are necessary for implementation
#and should be injected into the function
def make_gs_maps (model_gs, slide, wsi_map_bin_gs, p_s, m_p_s):
    #Define counters for gleason patterns for final score of the whole image
    counter_3 = 0.01
    counter_4 = 0.01
    counter_5 = 0.01
    counter_p = 0
    gp3 = 0.01
    gp4 = 0.01
    gp5 = 0.01
    
    #generate chunk numpy array for Gleason scoring based on the mathematical binary map size
    wsi_map_gs_preds = np.zeros((wsi_map_bin_gs.shape[0], wsi_map_bin_gs.shape[1], 3) , dtype=np.float32)
    
    #analyse slide, but only the tumor patches
    for he in range(wsi_map_bin_gs.shape[0]):
        for wi in range(wsi_map_bin_gs.shape[1]):
            if wsi_map_bin_gs [he,wi] == 9999: # contains tumor
                #Update Counter for number of tumor patches
                counter_p = counter_p + 1
                #Grade the patch and retrieve the WHO/ISUP grade group, raw predictions
                #and Gleason score for separate array
                #grade() is main function for generation of predictions and their
                #transformation to WHO/ISUP group and Gleason Score
                isup, preds, score = grade (model_gs, slide, p_s, m_p_s, he, wi)
                # isup = WHO/ISUP grade group codes: ISUP1 - 9991, ISUP2 - 9992, usw.

                #Update the Gleason score maps (both classes and raw predictions)
                wsi_map_bin_gs [he, wi] = isup
                wsi_map_gs_preds [he, wi] = preds
                
                #GS counter update
                counter_3 = counter_3 + score[0]
                counter_4 = counter_4 + score[1]
                counter_5 = counter_5 + score[2]
     
    #Calculate Gleason pattern percentages for the whole slide
    gp3 = counter_3 / (counter_3 + counter_4 + counter_5) * 100
    gp4 = counter_4 / (counter_3 + counter_4 + counter_5) * 100
    gp5 = counter_5 / (counter_3 + counter_4 + counter_5) * 100
    
    gp3 = round (gp3, 1)
    gp4 = round (gp4, 1)
    gp5 = round (gp5, 1)

    #Prepare to return
    score = (gp3, gp4, gp5)

    #Return map,
    return wsi_map_bin_gs, wsi_map_gs_preds, score, counter_p


#FUNCTION for grading single patches via Model type 2
def grade (model_gs, slide, p_s, m_p_s, he, wi):
    patch = slide.read_region((wi*600, he*600), 0, (p_s,p_s)).convert('RGB')
    
    patch = patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)
    #IMAGE TO ARRAY, PREPROCESSING
    patch = image.img_to_array(patch)
    patch = standardizer.transform(patch)
    patch = stain_norm.transform(patch)
    patch = np.float32(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model_gs.predict(patch)
    gp_3 = round(preds [0,0] * 100, 1)
    gp_4 = round(preds [0,1] * 100, 1)
    gp_5 = round(preds [0,2] * 100, 1)
    score = (gp_3, gp_4, gp_5)
    #gscoring() is a function to transform percentages of individual patterns
    #into Gleason Score (primary and secondary pattern) and WHO/ISUP grade group
    #based on radical prostatectomy rules
    gs_1, gs_2, isup = gscoring(score)

    #return
    return isup, preds, score

#FUNCTION to transform percentages of individual patterns
#into Gleason Score (primary and secondary pattern) and WHO/ISUP grade group
#based on radical prostatectomy rules
def gscoring (score):
    
    gp3, gp4, gp5 = score
    
    if gp5 > 5: #Firstly, check if enough GP5 is present (>5%)
        if gp4 > 5: # Calculate if there is enough GP4 to be primary or secondary pattern
            if gp4 > gp5: # X+5, GS 3+5 is still possible
                if gp4 >= gp3: # 4+5
                    prim = 4
                    second = 5
                    ISUP = 9995
                else: # 3+5
                    prim = 3
                    second = 5
                    ISUP = 9994
            else: # 5+X (gp4 > 5, gp5 > 5, gp3 - not defined)
                if gp4 >= gp3: #5+4
                    prim = 5
                    second = 4
                    ISUP = 9995
                else: 
                    prim = 5 #5+3, if gp4 > 5 and gp4 !>= gp3, then gp3 > 5
                    second = 3
                    ISUP = 9994
        elif gp3 > 5: # In the absence of enough GP4; calculate if there is enough GP3 to be primary or secondary pattern
            if gp3 > gp5: # 3+5
                prim = 3
                second = 5
                ISUP = 9994
            else: # 5+3
                prim = 5
                second = 3
                ISUP = 9994
        else: #Both GP3 and GP4 are low (<5%)
            prim = 5 # 5+5
            second = 5
            ISUP = 9995
 
    else: #GP5 is not present at >5%, therefore GS does not include GP5
        if gp3 > ((100-gp5) / 2): #GP3 is dominant
            prim = 3
            if gp4 > 5: #calculate secondary pattern in 3+X
                second = 4 # 3+4
                ISUP = 9992
            else:
                second = 3
                ISUP = 9991 # 3+3
        else: #GP4 is dominant
            prim = 4
            if (gp3 > 5): #calculate secondary pattern in 4+X
                second = 3 # 4+3
                ISUP = 9993
            else:
                second = 4
                ISUP = 9994 # 4+4
    
    #return Gleason Score
    gs = (prim, second, ISUP)
    return gs

