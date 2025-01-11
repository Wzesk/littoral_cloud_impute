## https://github.com/NASA-IMPACT/Prithvi-EO-2.0
## https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo
## https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/inference.py
## https://ml-for-rs.github.io/iclr2024/camera_ready/papers/61.pdf


# imports go here
import omnicloudmask
from omnicloudmask import (
    predict_from_load_func,
    predict_from_array,
    load_s2,
)
import numpy as np
from PIL import Image

##this is just a stub, all code remains to be added
class cloud_impution:
    def __init__(self,path="/prithvi_params"):
        self.path = path
        self.yml_path = path + "/data.yml"
        self.weights_path = path + "/best.pt"

        # currenty setup to use an rgb image with a nir band.
        # these internal image objects will all be np.arrays
        self.rgb = None
        self.nir = None

        #store the mask
        self.cld = None
        self.usable_pixels = -1
    
    # todo...check if  the standard s2 cloudmask outperforms omnicloudmask when there is no nir band.
    # if so, check for nir, if not, import and use s2 cloudmask
    def mask_clouds(self, rgb_img, nir_img=None):
      #convert rgb to array
      self.rgb = np.array(rgb_img) 

      #if nir image is not provided, just use the rgb image
      #this will end up using the red band instead of nir.  not as good, but will work
      if nir_img is None:
        nir_img = rgb_img
        print("NIR image not provided, replacing with red")

      #convert nir to array
      self.nir = np.array(nir_img)

      # create array with red, green and NIR
      img_arr = np.dstack([self.rgb[:,:,0], self.rgb[:,:,1], self.nir[:,:,0]])

      #reshape array to be bands, height, width
      img_arr = np.moveaxis(img_arr, -1, 0)
      
      pred_mask = predict_from_array(img_arr)

      #merge cloud mask with cloud shadow mask
      pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
      pred_array = np.where(pred_array > 0, 1, 0)

      cld_pred = np.squeeze(pred_array)
      self.cld = Image.fromarray(cld_pred.astype(np.uint8)*255)

      self.usable_pixels = np.sum(cld_pred) / cld_pred.size

      return self.cld, self.usable_pixels


    def predict(self,model):
      """
      notes:
      this will impute pixels for rgb and nir.  Prithvi is designed to
      operate on multiband tifs so the rgb and nir can be reassembled into
      a 4 band np.array to create a familiar input

      model: the prithvi (or alt) model to use

      returns: imputed rgb and nir images
      """
      #format into 4 band array
      input_bands = np.dstack([self.rgb[:,:,0], self.rgb[:,:,1], self.rgb[:,:,2], self.nir[:,:,0]])

      #run impution, nothing is happening just setting to input
      imputed_bands = input_bands # model(input_bands)

      #reformat into two 3 band images
      self.imp_rgb = imputed_bands[:,:,:3]
      self.imp_nir = np.dstack([imputed_bands[:,:,3],imputed_bands[:,:,3],imputed_bands[:,:,3]])

      return "I am a stub, doing nothing right now..."
    

    def train_impution(self,epochs=100,imgsz=640,batch=8,mask_ratio=4,name='island_prithvi'):
        results = "just a stub, nothing happening yet"
        # model = prithvi(self.model_name)
        # results = model.train(
        # data=self.yaml,
        # imgsz=imgsz,
        # epochs=epochs,
        # batch=batch,
        # seed=7,
        # name=name,
        # mask_ratio=mask_ratio
        # )
        return results