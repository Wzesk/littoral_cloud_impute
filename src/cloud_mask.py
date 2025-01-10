"""Cloud mask detection module for satellite imagery preprocessing.

This module provides a lightweight preprocessing step for cloud detection in satellite imagery
using RGB and NIR bands. While more sophisticated models like Prithvi-EO-2.0 can perform
direct cloud imputation, this module serves as a quick pre-screening tool to:

1. Assess image quality before deeper processing
2. Calculate usable pixel ratios for scene selection
3. Identify regions requiring imputation
4. Optimize processing time by filtering heavily clouded scenes

The module utilizes the omnicloudmask library for efficient binary mask generation.
While potentially redundant with direct imputation approaches, it offers a fast
initial assessment that can inform subsequent processing decisions.

Dependencies:
-----------
omnicloudmask : package
    Cloud detection model
numpy : package
    Array operations
PIL : package
    Image processing
"""

from typing import Tuple

import numpy as np
from omnicloudmask import predict_from_array
from PIL import Image


def pred_clouds_from_rgbnir(rgb_img: Image.Image, nir_img: Image.Image) -> Tuple[Image.Image, float]:
    """Predict cloud and cloud shadow masks from RGB and NIR images.

    This function takes RGB and NIR images as input and returns a binary mask
    indicating cloud/shadow presence (255 for cloud/shadow, 0 for clear) along with
    the fraction of usable (clear) pixels.

    Parameters:
    ----------
    rgb_img : PIL.Image.Image
        RGB image as a PIL Image object
    nir_img : PIL.Image.Image
        NIR image as a PIL Image object

    Returns:
    -------
    mask : PIL.Image.Image
        Binary mask where 255 indicates cloud/shadow presence
    usable : float
        Fraction of usable (clear) pixels in the image

    Examples:
    --------
    >>> from PIL import Image
    >>> rgb_img = Image.open('rgb.tif')
    >>> nir_img = Image.open('nir.tif')
    >>> mask, usable = pred_clouds_from_rgbnir(rgb_img, nir_img)
    >>> mask.save('cloud_mask.png')
    >>> print(f"Usable pixels: {usable:.2%}")

    Notes:
    -----
    This function serves as a preprocessing step to quickly assess image quality
    before more computationally intensive operations. While models like Prithvi-EO-2.0
    can perform direct cloud imputation, this quick assessment helps optimize the
    processing pipeline by identifying which scenes warrant further processing.
    """
    # Convert rgb and nir images to arrays
    rgb_arr = np.array(rgb_img)
    nir_arr = np.array(nir_img)

    # Create array with red, green and NIR
    img_arr = np.dstack([rgb_arr[:, :, 0], rgb_arr[:, :, 1], nir_arr[:, :, 0]])

    pred_mask = predict_from_array(img_arr)

    # Merge cloud mask with cloud shadow mask
    pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)

    cld_pred = np.squeeze(pred_array)
    cld_msk = Image.fromarray(cld_pred.astype(np.uint8) * 255)

    usable_pixels = np.sum(cld_pred) / cld_pred.size

    return cld_msk, usable_pixels
