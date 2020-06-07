#EXTRACTION OF META-DATA FROM WHOLE-SLIDE IMAGE
#Might be neccesary to optimize extraction for WSI scanned by different vendors
#due to different construction / coding of metadata

from PIL import Image


def slide_info (slide, p_s):
    #Objective power
    obj_power = slide.properties ["openslide.objective-power"]
    
    #Microne per pixel
    mpp = slide.properties ["openslide.mpp-x"]
    
    #Vendor
    vendor = slide.properties ["openslide.vendor"]
    
    #Extract and save dimensions of level 0 of the pyramide
    dim_l0 = slide.level_dimensions[0]
    w_l0 = dim_l0 [0]
    h_l0 = dim_l0 [1]
    
    #Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)
    
    #Output BASIC DATA
    print ("")
    print ("Basic data about processed whole-slide image")
    print ("")
    print ("Vendor: ", vendor)
    print ("Scan magnification: ", obj_power)
    print ("Microns per pixel:", mpp)
    print ("Height: ", h_l0)
    print ("Width: ", w_l0)
    print ("Patch size: ", p_s, "x", p_s)
    print ("Width: number of patches: ", patch_n_w_l0)
    print ("Height: number of patches: ", patch_n_h_l0)
    print ("Number of patches to process: ", patch_n_w_l0 * patch_n_h_l0)
    
    thumbnail = slide.associated_images ["thumbnail"]
    thumbnail = thumbnail.resize((patch_n_w_l0*3,patch_n_h_l0*3), Image.ANTIALIAS)
    
    #return(patch_n_w_l0, patch_n_h_l0)
    return(thumbnail, patch_n_w_l0, patch_n_h_l0)
