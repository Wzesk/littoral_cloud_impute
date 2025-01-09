"""Cloud mask detection module for satellite imagery using RGB and NIR bands.

This module provides functionality for detecting clouds in satellite imagery using RGB
and NIR bands. It utilizes the omnicloudmask library to generate cloud and shadow
masks from multispectral imagery.

Dependencies:
    - omnicloudmask: Cloud detection model
    - numpy: Array operations
    - PIL: Image processing
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

    Args:
        rgb_img: RGB image as a PIL Image object
        nir_img: NIR image as a PIL Image object

    Returns:
        A tuple containing:
            - Binary mask where 255 indicates cloud/shadow presence
            - Fraction of usable (clear) pixels in the image

    Example:
        >>> rgb_img = Image.open('rgb.tif')
        >>> nir_img = Image.open('nir.tif')
        >>> mask, usable = pred_clouds_from_rgbnir(rgb_img, nir_img)
        >>> mask.save('cloud_mask.png')
        >>> print(f"Usable pixels: {usable:.2%}")
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
