# References

## Goal 1: Cloud Detection and Masking

### Primary Methods
- **OmniCloudMask**
  - Repository: [OmniCloudMask](https://github.com/allenai/omnicloudmask)
  - Paper: "OmniCloudMask: A Cloud Masking Model for Multi-Sensor Multi-Platform Remote Sensing Data"

### Related Research
- **Fmask Algorithm**
  - Paper: "Fmask 4.0: Improved cloud and cloud shadow detection in Landsats 4–8 and Sentinel-2 imagery" (2019)
  - DOI: 10.1016/j.rse.2019.05.024
  - Focus: Multi-sensor cloud detection

- **S2Cloudless**
  - Repository: [Sentinel Hub](https://github.com/sentinel-hub/sentinel2-cloud-detector)
  - Paper: "Sentinel-2 Cloud Detection using Machine Learning" (2020)
  - Application: Sentinel-2 specific cloud detection

## Goal 2: Cloud Imputation and Data Fusion

### Primary Methods
- **Prithvi-EO-2.0**
  - Repository: [NASA-IMPACT/Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
  - Demo: [Hugging Face Space](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo)
  - Model: [Hugging Face Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
  - Paper: [ML4RS @ ICLR 2024](https://ml-for-rs.github.io/iclr2024/camera_ready/papers/61.pdf)

### Related Research
- **Multi-temporal Fusion**
  - Paper: "Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion" (2020)
  - DOI: 10.1016/j.isprsjprs.2020.06.023
  - Focus: SAR-optical fusion for cloud removal

- **GAN-based Approaches**
  - Paper: "Cloud-GAN: Cloud Removal for Sentinel-2 Imagery Using a Cyclic Consistent Generative Adversarial Networks" (2019)
  - DOI: 10.1109/IGARSS.2019.8898776
  - Focus: GAN-based cloud removal

## Goal 3: Shoreline Superresolution

### Relevant Methods
- **CoastSat**
  - Repository: [CoastSat](https://github.com/kvos/CoastSat)
  - Paper: "CoastSat: A Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery" (2019)
  - DOI: 10.1016/j.envsoft.2019.104528

- **Shoreline Detection**
  - Paper: "Automated Extraction of Shorelines from Satellite Imagery: A Review" (2021)
  - DOI: 10.3390/rs13193930
  - Focus: State-of-the-art review

### Superresolution Techniques
- **SRGAN for Remote Sensing**
  - Paper: "Super-Resolution of Sentinel-2 Images Using Deep Learning" (2020)
  - DOI: 10.3390/rs12152207
  - Focus: Satellite image superresolution

## Data Sources

### Primary Sources
- Sentinel-2 MSI: [ESA Sentinel Online](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
- Landsat Collection 2: [USGS Landsat](https://www.usgs.gov/landsat-missions)

### Additional Resources
- Google Earth Engine: [Platform](https://earthengine.google.com/)
  - Provides access to multiple satellite datasets
  - Enables large-scale processing

## Implementation Resources

### Cloud Computing
- Microsoft Planetary Computer: [Documentation](https://planetarycomputer.microsoft.com/docs)
- Google Earth Engine: [Developer Resources](https://developers.google.com/earth-engine)

### Deep Learning Frameworks
- PyTorch: [Documentation](https://pytorch.org/docs)
- Hugging Face Transformers: [Documentation](https://huggingface.co/docs)
