#IMPORT LIBRARIES
import numpy as np
from wsi_heatmaps import gen_heatmaps, gen_heatmaps_gs
from keras.preprocessing import image
from PIL import Image
import cv2

#RECORD FUNCTION FOR PREDS MAPS
#function to automatize saving of the individual predictions into
#numpy array chunk for slide map
def record_map_preds (wsi_map_preds, hi, wi, preds):
    wsi_map_preds [hi,wi] = preds


#Function to MAKE MATHEMATICAL BINARY MAP based on individual predictions
#CODES: Background = 0, BENIGN = 1111, TUMOR = 9999
def make_wsi_map_bin (wsi_map_preds, patch_n_w_l0, patch_n_h_l0):
    #make numpy chunk for map based on slide size
    wsi_map_bin = np.zeros((patch_n_h_l0, patch_n_w_l0, 1) , dtype=np.int16)
    #loop for automatic procession of patch cells in map
    for he in range(wsi_map_preds.shape[0]):
        for wi in range(wsi_map_preds.shape[1]):
            if sum(wsi_map_preds [he,wi,0:3]) > 0:
                if wsi_map_preds [he,wi,2] > 0.5:
                    entity = 9999 # tumor
                else:
                    entity = 1111 # normal
            else:
                entity = 0 # background without tissue
            wsi_map_bin [he,wi] = entity
    return wsi_map_bin


#Function to MAKE HEATMAP AS IMAGE, OUTPUT in NUMPY ARRAY FORM
#hmap_p_s is a heatmap patch size (downsizing to prevent large file sizes)
def make_wsi_heatmap (wsi_map_bin, hmap_p_s):
    #generate heatmap chung images as numpy arrays
    heatmap_tumor, heatmap_normal, blank_patch = gen_heatmaps(hmap_p_s)
    heatmap_blank = np.uint8(image.img_to_array(blank_patch))
    #loop processing of the whole map
    #creation of heatmap image as numpy array according to prior classification results
    #through concatenation of individual patches
    for he in range(wsi_map_bin.shape[0]):
            for wi in range(wsi_map_bin.shape[1]):
                if wsi_map_bin [he,wi] == 0:
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 1111:
                    heatmap = heatmap_normal
                else:
                    heatmap = heatmap_tumor
                
                if (wi==0):
                    temp_image = heatmap
                else:
                    temp_image = np.concatenate((temp_image, heatmap), axis=1)
           
            if (he==0):
                end_image = temp_image
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
            
            del temp_image
    #return map as image in numpy array form (end_image)
    return(end_image)

###SIMILAR PRINCIPLE TO GENERATE GLEASON SCORE HEATMAP as image
#MAKE HEATMAP AS IMAGE for Gleason Score (as numpy array)
def make_wsi_heatmap_gs (wsi_map_gs_bin, hmap_p_s):
    #generate heatmaps tumor / benign / background
    heatmap_tumor, heatmap_normal, blank_patch = gen_heatmaps(hmap_p_s)
    heatmap_blank = np.uint8(image.img_to_array(blank_patch))
    #generate heatmaps for ISUP groups
    heatmap_1, heatmap_2, heatmap_3, heatmap_4, heatmap_5 = gen_heatmaps_gs(hmap_p_s)
    
    for he in range(wsi_map_gs_bin.shape[0]):
            for wi in range(wsi_map_gs_bin.shape[1]):
                if wsi_map_gs_bin [he,wi] == 0:
                    heatmap = heatmap_blank
                elif wsi_map_gs_bin [he,wi] == 1111:
                    heatmap = heatmap_normal
                else:
                    if wsi_map_gs_bin [he,wi] == 9991:
                        heatmap = heatmap_1
                    if wsi_map_gs_bin [he,wi] == 9992:
                        heatmap = heatmap_2
                    if wsi_map_gs_bin [he,wi] == 9993:
                        heatmap = heatmap_3
                    if wsi_map_gs_bin [he,wi] == 9994:
                        heatmap = heatmap_4
                    if wsi_map_gs_bin [he,wi] == 9995:
                        heatmap = heatmap_5
                
                if (wi==0):
                    temp_image = heatmap
                else:
                    temp_image = np.concatenate((temp_image, heatmap), axis=1)
           
            if (he==0):
                end_image = temp_image
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
            
            del temp_image
    return(end_image)



#Function to MAKE OVERLAID IMAGE: HEATMAP ON REDUCED AND CROPPED WHOLE-SLIDE IMAGE
#slide per se, heatmap image, slide sizes and size reduction factor are necessary
#inputs to implement this action
def make_overlay (slide, wsi_heatmap_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor):

    #retrieve slide dimensions on tle level 0 of the pyramide
    w_l0, h_l0 = slide.level_dimensions[0]

    #get reduced version of the whole image
    slide_reduced = slide.get_thumbnail((w_l0/overlay_factor,h_l0/overlay_factor))

    #preparing to crop some background of the slide to
    #ideally overlay heatmap
    hei = patch_n_h_l0 * p_s / overlay_factor
    wid = patch_n_w_l0 * p_s / overlay_factor
    #cropping the whole slide image
    area = (0,0, wid, hei)
    slide_reduced_crop = slide_reduced.crop(area)

    #overlay using opencv
    heatmap_temp = wsi_heatmap_im.resize(slide_reduced_crop.size, Image.ANTIALIAS)          
    overlay = cv2.addWeighted(np.array(slide_reduced_crop),0.8,np.array(heatmap_temp),0.2,0)
    return (overlay)
