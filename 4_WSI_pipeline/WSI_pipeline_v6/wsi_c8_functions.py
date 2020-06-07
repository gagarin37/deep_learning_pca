from PIL import Image, ImageOps
from statistics import median
import numpy as np
from keras.preprocessing import image

#Function for processing of predictions for 8 single patch derivates
def gateway_median (model, patch):
    #native version
    base = patch #1
    #rotations
    r90 = patch.rotate(90) #2
    r180 = patch.rotate(180) #3
    r270 = patch.rotate(270) #4
    #flip/rotations
    r90_VF = ImageOps.flip(r90) #5
    r270_VF = ImageOps.flip(r270) #6
    #flips
    VF = ImageOps.flip(base) #7
    HF = base.transpose(Image.FLIP_LEFT_RIGHT) #8
    #calculate final predictions as median
    pred_stack = np.vstack((pred(model, base),
                            pred(model, r90),
                            pred(model, r180),
                            pred(model, r270),
                            pred(model, r90_VF),
                            pred(model, r270_VF),
                            pred(model, VF),
                            pred(model, HF)))
    pred_1 = median(pred_stack[0:8,0])
    pred_2 = median(pred_stack[0:8,1])
    pred_3 = median(pred_stack[0:8,2])
    preds_med = np.array([pred_1, pred_2, pred_3])
    return preds_med

#Function for generation of prediction of single patch derivates
def pred (model, patch):
    #IMAGE TO ARRAY, PREPROCESSING
    patch = image.img_to_array(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    #return predictions
    return preds
###############################