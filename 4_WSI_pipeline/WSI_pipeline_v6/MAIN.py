#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""

# =============================================================================
# 0. SET PARAMETERS
# =============================================================================

###PATCH SIZE
p_s = 600
#MODEL PATCH SIZE
m_p_s = 350      # analaysis at 23x magnification
#m_p_s = 300     # analysis at 20x magnification
#HEATMAP (BINARY MAP) PATCH SIZE
hmap_p_s = 10
###PATH_TO_SLIDE (to be defined)
slide_dir = '' #path to WSI file
slide_name = ''
###MODEL PATH Tumor vs Benign (Type 1) (to be defined)
model_tvn_dir = ''  #path to model folder
model_tvn_name = '' #h5-file of model
###MODEL PATH Gleason grading (Type 2) (to be defined)
model_gs_dir = ''  #path to Gleason grading model
model_gs_name = '' #h5-file of model
###OVERLAY FACTOR: REDUCTION OF ORIGINAL SLIDE SIZE FOR OVELAID IMAGES
overlay_factor = 10
###FLAG: apply C8 strategy for analysis?
c8_flag = True
###FLAG: investigate environment of single patches?
env_flag = True
###FLAG: make Gleason scoring?
gs_flag = True


# =============================================================================
# 1. LIBRARIES
# =============================================================================
from keras.models import load_model
from openslide import open_slide
from PIL import Image
import os
from wsi_slide_info import slide_info
from wsi_process import slide_process, slide_process_C8
from wsi_maps import make_wsi_map_bin, make_wsi_heatmap, make_overlay, make_wsi_heatmap_gs
import numpy as np
from wsi_single_env import check_single_environment
from wsi_gleason import make_gs_maps
import copy



# =============================================================================
# 2. LOAD MODELs
# =============================================================================
path_model_tvn = os.path.join(model_tvn_dir, model_tvn_name)
model_tvn = load_model(path_model_tvn)
model_tvn.summary()

path_model_gs = os.path.join(model_gs_dir, model_gs_name)
model_gs = load_model(path_model_gs)
model_gs.summary()

# =============================================================================
# 3. OPEN/PROCESS WSI, EXTRACT DATA, PRESENT BASIC DATA ABOUT SLIDE
# =============================================================================
#Open slide
path_slide = os.path.join(slide_dir, slide_name)
slide = open_slide(path_slide)
###Print Meta-Data, Calculate number of patches (width and height), generate thumbnail
thumbnail, patch_n_w_l0, patch_n_h_l0 = slide_info(slide, p_s)
#Show thumbnail
thumbnail


# =============================================================================
# 4. WSI PROCESSING TO GENERATE MATHEMATICAL MAP WITH PREDICTIONS
# =============================================================================
#Returns mathematical map with background/tissue detection and
#raw predictions for tumor vs benign tissue classification
if c8_flag == True: #Implement C8 strategy (s. Methods for description)
    wsi_map_preds = slide_process_C8 (model_tvn, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s)
else: #then implement conventional C1 strategy (for every single patch only
      #native version would be analysed)
    wsi_map_preds = slide_process (model_tvn, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s)


# =============================================================================
# 5. MAKE MATHEMATICAL BINARY MAP FROM MATHEMATICAL MAP WITH RAW PREDICTIONS
# =============================================================================
#Returns binary mathematical map with codes for classes
#(CODES: background = 0, benign = 1111, tumor = 9999)
#Processing of mathematical map with predictions
#Threshold for classification is always 0.5
wsi_map_bin = make_wsi_map_bin (wsi_map_preds, patch_n_w_l0, patch_n_h_l0)


# =============================================================================
# 6. SAVE MATHEMATICAL MAPS FOR FURTHER USE
# =============================================================================
#Save mathematical map (predictions)
wsi_map_preds_name = "output/" + slide_name + "_map_C8_preds.npy"
np.save(wsi_map_preds_name, wsi_map_preds)
#Save mathematical map (binary/classes)
wsi_map_bin_name = "output/" + slide_name + "_map_C8_bin.npy"
np.save(wsi_map_bin_name, wsi_map_bin)


# =============================================================================
# 7. MAKE AND SAVE HEATMAP (as image) FROM MATHEMATICAL BINARY MAP (for C8 only)
# =============================================================================

wsi_heatmap = make_wsi_heatmap (wsi_map_bin, hmap_p_s)
                    
#Save WSI HEATMAP as image file
#wsi_heatmap = np.uint8(wsi_heatmap)
wsi_heatmap_im = Image.fromarray(wsi_heatmap)
wsi_heatmap_im_name = "output/" + slide_name + "_heatmap_C8.png"
wsi_heatmap_im.save(wsi_heatmap_im_name)

# =============================================================================
# 8. MAKE AND SAVE OVERLAID IMAGE (for C8 only): HEATMAP OVERLAID ON REDUCED
# AND CROPPED SLIDE IMAGE
# =============================================================================

#input = WSI HEATMAP image
#overlay_factor is a factor of size reduction of whole slide images
#other parameters are necessary for implementation
overlay = make_overlay (slide, wsi_heatmap_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor)

#Save overlaid image
overlay_im = Image.fromarray(overlay)
overlay_im_name = "output/" + slide_name + "_overlay_C8.png"
overlay_im.save(overlay_im_name)

# =============================================================================
# 9. CHECK SINGLE POSITIVE PATCHES ("ENVIRONMENT" CHECK) s. Methods for details
# =============================================================================

#Input = mathematical binary map with class codes
if env_flag == True:
    wsi_map_bin_single = copy.copy(wsi_map_bin)
    wsi_map_bin_single = check_single_environment (model_tvn, slide, wsi_map_bin_single, p_s, m_p_s)
#Output = optimized mathematical binary map with class codes

#Save generated optimized mathematical binary map for further use
wsi_map_bin_single_name = "output/" + slide_name + "_map_C8_bin_SINGLE.npy"
np.save(wsi_map_bin_single_name, wsi_map_bin_single)


###Further steps 10 and 11 are similar to steps performed after implementation of C8 strategy above,
###but not for a map optimized through Environment Analysis of single patches
# =============================================================================
# 10. MAKE AND SAVE HEATMAP as image FROM MATHEMATICAL BINARY MAP (now for C8_SINGLE)
# =============================================================================
#MAKE HEATMAP
wsi_heatmap_single = make_wsi_heatmap (wsi_map_bin_single, hmap_p_s)
                    
#Save WSI HEATMAP as image
#wsi_heatmap_single = np.uint8(wsi_heatmap_single)
wsi_heatmap_single_im = Image.fromarray(wsi_heatmap_single)
wsi_heatmap_single_im_name = "output/" + slide_name + "_heatmap_C8_SINGLE.png"
wsi_heatmap_single_im.save(wsi_heatmap_single_im_name)

# =============================================================================
# 11. MAKE AND SAVE OVERLAY (now for C8_SINGLE): HEATMAP ON REDUCED AND CROPPED SLIDE CLON
# =============================================================================
overlay_single = make_overlay (slide, wsi_heatmap_single_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor)

#Save overlaid image
overlay_single_im = Image.fromarray(overlay_single)
overlay_single_im_name = "output/" + slide_name + "_overlay_C8_SINGLE.png"
overlay_single_im.save(overlay_single_im_name)


# =============================================================================
# 12. MAKE AND SAVE GLEASON SCORE MATHEMATICAL MAPS, HEATMAP AND OVERLAID IMAGE
# =============================================================================
if gs_flag == True:
    #create a hard copy of mathematical binary map of the slide (background, tumor, benign)
    #to further directly update it according to grading results
    wsi_map_bin_temp = copy.copy(wsi_map_bin_single)
    wsi_map_gs_bin, wsi_map_gs_pred, score, tu_num_patch = make_gs_maps (model_gs, slide, wsi_map_bin_temp, p_s, m_p_s)

    #Save mathemtical PREDICTIONS maps for GS
    wsi_map_gs_pred_name = "output/" + slide_name + "_map_C8_SINGLE_GS_preds.npy"
    np.save(wsi_map_gs_pred_name, wsi_map_gs_pred)
    
    #Save mathematical BINARY map for GS
    wsi_map_gs_bin_name = "output/" + slide_name + "_map_C8_SINGLE_GS_bin.npy"
    np.save(wsi_map_gs_bin_name, wsi_map_gs_bin)

    #Optional: show gleason score, number of patches with tumor

    # =============================================================================
    # 13. MAKE AND SAVE HEATMAP as image FROM MATHEMATICAL BINARY GLEASON SCORE MAP
    # =============================================================================
    #Make heatmap as image
    wsi_heatmap_gs = make_wsi_heatmap_gs (wsi_map_gs_bin, hmap_p_s)
    
    #Save WSI Gleason Score HEATMAP as image file
    #wsi_heatmap_single = np.uint8(wsi_heatmap_single)
    wsi_heatmap_gs_im = Image.fromarray(wsi_heatmap_gs)
    wsi_heatmap_gs_im_name = "output/" + slide_name + "_heatmap_C8_SINGLE_GS.png"
    wsi_heatmap_gs_im.save(wsi_heatmap_gs_im_name)
    
    # =============================================================================
    # 14. MAKE AND SAVE OVERLAID IMAGE: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
    # =============================================================================
    overlay_gs = make_overlay (slide, wsi_heatmap_gs_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor)
    
    #Save overlaid image
    overlay_gs_im = Image.fromarray(overlay_gs)
    overlay_gs_im_name = "output/" + slide_name + "_overlay_C8_SINGLE_GS.png"
    overlay_gs_im.save(overlay_gs_im_name)



###END OF THE SCRIPT