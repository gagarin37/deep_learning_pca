#LOAD HEATMAP CHUNK IMAGES

from keras.preprocessing import image
import numpy as np

def gen_heatmaps (m_p_s):
    #tumor
    heatmap_tumor = image.load_img("images/heatmap_tumor.jpg", target_size=(m_p_s, m_p_s))
    heatmap_tumor = np.uint8(image.img_to_array(heatmap_tumor))
    
    #normal
    heatmap_normal = image.load_img("images/heatmap_normal.jpg", target_size=(m_p_s, m_p_s))
    heatmap_normal = np.uint8(image.img_to_array(heatmap_normal))
    
    #blank
    blank_patch = image.load_img("images/blank_patch.jpg", target_size=(m_p_s, m_p_s))
    
    return (heatmap_tumor, heatmap_normal, blank_patch)

def gen_heatmaps_gs (m_p_s):
    
    heatmap_1 = image.load_img("images/1.jpg", target_size=(m_p_s, m_p_s))
    heatmap_1 = np.uint8(image.img_to_array(heatmap_1))
    
    heatmap_2 = image.load_img("images/2.jpg", target_size=(m_p_s, m_p_s))
    heatmap_2 = np.uint8(image.img_to_array(heatmap_2))
    
    heatmap_3 = image.load_img("images/3.jpg", target_size=(m_p_s, m_p_s))
    heatmap_3 = np.uint8(image.img_to_array(heatmap_3))
    
    heatmap_4 = image.load_img("images/4.jpg", target_size=(m_p_s, m_p_s))
    heatmap_4 = np.uint8(image.img_to_array(heatmap_4))
    
    heatmap_5 = image.load_img("images/5.jpg", target_size=(m_p_s, m_p_s))
    heatmap_5 = np.uint8(image.img_to_array(heatmap_5))
    
    return (heatmap_1, heatmap_2, heatmap_3, heatmap_4, heatmap_5)