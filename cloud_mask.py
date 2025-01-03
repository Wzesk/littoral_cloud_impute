
import omnicloudmask
from omnicloudmask import (
    predict_from_load_func,
    predict_from_array,
    load_s2,
)
import numpy as np
from PIL import Image

def pred_clouds_from_rgbnir(self, rgb_img, nir_img):
    #convert rgb and nir images to arrays
    rgb_arr = np.array(rgb_img) 
    nir_arr = np.array(nir_img)

    # create array with red, green and NIR
    img_arr = np.dstack([rgb_arr[:,:,0], rgb_arr[:,:,1], nir_arr[:,:,0]])
    
    pred_mask = predict_from_array(img_arr)

    #merge cloud mask with cloud shadow mask
    pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)

    cld_pred = np.squeeze(pred_array)
    cld_msk = Image.fromarray(cld_pred.astype(np.uint8)*255)

    usable_pixels = np.sum(cld_pred) / cld_pred.size

    return cld_msk, usable_pixels