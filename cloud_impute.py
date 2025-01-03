## https://github.com/NASA-IMPACT/Prithvi-EO-2.0
## https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo
## https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/inference.py
## https://ml-for-rs.github.io/iclr2024/camera_ready/papers/61.pdf


# imports go here


##this is just a stub, all code remains to be added
class cloud_impution:
    def __init__(self,path="/prithvi_params"):
        self.path = path
        self.yml_path = path + "/data.yml"
        self.weights_path = path + "/best.pt"
    

    def train(self,epochs=100,imgsz=640,batch=8,mask_ratio=4,name='island_prithvi'):
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
    
      def predict(self,image,model):
        results = "just a stub, nothing happening yet"
        #results = model(image)
        return results