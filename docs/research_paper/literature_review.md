# Literature Review

## 1. Cloud Detection and Masking Methods

The detection and masking of clouds in satellite imagery represents a fundamental challenge in remote sensing, particularly for coastal regions where water-land interfaces create complex spectral signatures. Recent advances in deep learning and multi-sensor approaches have significantly improved detection accuracy and generalization capabilities. This section examines key methodologies that address the challenges of cloud detection in coastal environments.

### Primary Methods

1. **OmniCloudMask**
   - **Repository:** [OmniCloudMask](https://github.com/allenai/omnicloudmask)
   - **Paper:** *OmniCloudMask: A Cloud Masking Model for Multi-Sensor Multi-Platform Remote Sensing Data*
   - **Summary:** OmniCloudMask is a deep learning-based framework designed for cloud detection across multiple remote sensing platforms (e.g., Landsat, Sentinel). It employs a robust architecture capable of adapting to data from different sensors while maintaining consistent accuracy across diverse environments.
   - **Key Findings:** Achieves state-of-the-art performance on benchmark datasets and reduces false positives in high-albedo regions (e.g., snow or bright surfaces). Utilizes transfer learning for rapid adaptation to new datasets.
   - **Novel Contributions:** First large-scale, multi-sensor deep learning model for cloud masking that generalizes across platforms. Offers pre-trained weights for ease of integration.
   - **Relevance:** Its ability to handle multiple sensors aligns perfectly with your goal of detecting clouds using spatial context.
   - **Code:** Available on GitHub.

2. **CloudS2Mask**
   - **Repository:** [CloudS2Mask GitHub](https://github.com/DPIRD-DMA/CloudS2Mask)
   - **Paper:** *CloudS2Mask: A Novel Deep Learning Approach for Improved Cloud and Cloud Shadow Masking in Sentinel-2 Imagery*
     **Authors:** Nicholas Wright, John M.A. Duncan, J. Nik Callow, Sally E. Thompson, Richard J. George
     **Journal:** Remote Sensing of Environment
     **DOI:** [10.1016/j.rse.2024.114122](https://doi.org/10.1016/j.rse.2024.114122)
   - **Summary:** CloudS2Mask specializes in segmenting clouds and shadows in Sentinel-2 imagery. The model is designed to be computationally efficient, with the capability to process a Sentinel-2 scene in under three seconds on high-end GPUs. It supports adjustable accuracy settings for high-performance computing and non-GPU environments.
   - **Key Findings:** Benchmarked higher accuracy than operational algorithms like S2Cloudless, with precise shadow handling and negligible false negatives in thin clouds.
   - **Novel Contributions:** Introduces tunable settings for accuracy vs. processing speed, with CPU-only support. Includes extensive benchmarks on high-latitude and tropical datasets.
   - **Relevance:** A highly efficient, specialized tool for spatial cloud detection in Sentinel-2 imagery.
   - **Code:** Available on GitHub.

### Traditional Approaches

3. **Fmask Algorithm**
   - **Paper:** *Fmask 4.0: Improved Cloud and Cloud Shadow Detection in Landsats 4–8 and Sentinel-2 Imagery*
     **DOI:** [10.1016/j.rse.2019.05.024](https://doi.org/10.1016/j.rse.2019.05.024)
   - **Summary:** Fmask 4.0 refines its threshold-based algorithm to detect thin clouds and their shadows in Landsat and Sentinel-2 imagery. It incorporates dynamic thresholds based on scene-specific brightness values, improving detection consistency across diverse landscapes.
   - **Key Findings:** Enhanced ability to handle mixed pixels in cloud boundaries and significant improvements in cloud shadow detection, particularly for mountainous regions.
   - **Novel Contributions:** Integrates adjacency analysis to differentiate between snow and clouds, reducing false positives.
   - **Relevance:** Remains a reliable baseline for multi-sensor cloud detection.

4. **S2Cloudless**
   - **Repository:** [S2Cloudless on GitHub](https://github.com/sentinel-hub/sentinel2-cloud-detector)
   - **Paper:** *Sentinel-2 Cloud Detection Using Machine Learning*
     **DOI:** [10.1016/j.rse.2020.111818](https://doi.org/10.1016/j.rse.2020.111818)
   - **Summary:** A machine learning-based cloud detection tool optimized for Sentinel-2 imagery. S2Cloudless utilizes gradient boosting on manually labeled features (e.g., brightness, temperature) to identify cloud pixels efficiently.
   - **Key Findings:** Demonstrates lightweight computation with a competitive F1-score of 0.92 on benchmark datasets. Effective for large-scale applications with limited computational resources.
   - **Novel Contributions:** Designed specifically for operational deployment, balancing simplicity and accuracy.
   - **Relevance:** Ideal for rapid and lightweight cloud detection workflows.
   - **Code:** Available on GitHub.

### Comparative Analysis

| Method | Key Strengths | Technical Innovation | Performance | Limitations |
|--------|--------------|---------------------|-------------|-------------|
| OmniCloudMask | Multi-sensor support, Transfer learning | Cross-platform generalization | SOTA on benchmarks | High compute needs |
| CloudS2Mask | Fast processing, CPU support | Tunable accuracy-speed tradeoff | Better than S2Cloudless | Sentinel-2 only |
| Fmask 4.0 | Multi-sensor support | Dynamic thresholding | Good in mountains | Processing speed |
| S2Cloudless | Lightweight, Efficient | Gradient boosting | F1-score: 0.92 | Limited to Sentinel-2 |

## 2. Cloud Imputation and Data Fusion

Cloud imputation represents the critical second stage in our pipeline, focusing on reconstructing obscured areas using multiple data sources and advanced deep learning techniques. Recent advances in transformer architectures and multi-modal fusion have significantly improved reconstruction quality.

### Primary Methods

1. **Prithvi-EO-2.0**
   - **Repository:** [Prithvi-EO-2.0 GitHub](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
   - **Demo:** [Hugging Face Space](https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-EO-2.0-Demo)
   - **Paper:** *Prithvi-EO-2.0: Multi-Sensor Earth Observation Data Fusion with Transformer Models*
     **Authors:** NASA-IMPACT Team, ICLR 2024.
     **DOI:** [ML4RS Paper](https://ml-for-rs.github.io/iclr2024/camera_ready/papers/61.pdf)
   - **Summary:** A state-of-the-art transformer model designed for seamless fusion of multi-sensor Earth observation data, emphasizing noise robustness and spatial-temporal consistency. It performs imputation of cloud-contaminated pixels by leveraging attention-based fusion from other modalities.
   - **Key Findings:** Outperforms conventional CNN-based methods on multi-modal datasets, with an increase of up to 10% in PSNR for cloud imputation tasks.
   - **Novel Contributions:** First use of transformers for large-scale Earth observation data fusion, introducing dynamic context-aware imputation.
   - **Relevance:** Directly applicable to cloud removal and fusion goals.
   - **Code:** Available on GitHub.

2. **Multi-Temporal Fusion**
   - **Paper:** *Cloud Removal in Sentinel-2 Imagery Using a Deep Residual Neural Network and SAR-Optical Data Fusion*
     **DOI:** [10.1016/j.isprsjprs.2020.06.023](https://doi.org/10.1016/j.isprsjprs.2020.06.023)
   - **Summary:** Proposes a residual neural network that combines SAR backscatter data with optical images for cloud removal in Sentinel-2 datasets. This approach preserves spectral consistency while using temporal and multi-modal data fusion to improve robustness.
   - **Key Findings:** Achieves seamless fusion with SAR imagery, significantly improving scene continuity under dense cloud cover.
   - **Novel Contributions:** Introduces spectral-spatial consistency as a loss function.
   - **Relevance:** Demonstrates the utility of SAR data for cloud removal.

3. **Cloud-GAN**
   - **Paper:** *Cloud-GAN: Cloud Removal for Sentinel-2 Imagery Using Cyclic Consistent GANs*
     **DOI:** [10.1109/IGARSS.2019.8898776](https://doi.org/10.1109/IGARSS.2019.8898776)
   - **Summary:** Employs a cyclic consistent GAN architecture for reconstructing cloud-free images. Ensures high realism in reconstructed images while minimizing spectral distortions.
   - **Key Findings:** Yields realistic cloud-free imagery without requiring paired datasets.
   - **Novel Contributions:** First application of cycle consistency loss in cloud removal for satellite imagery.
   - **Relevance:** Demonstrates advanced deep learning applications for cloud suppression.

### Comparative Analysis

| Method | Architecture | Key Innovation | Performance Metrics | Code Availability |
|--------|-------------|----------------|-------------------|------------------|
| Prithvi-EO-2.0 | Transformer | Multi-modal fusion | 10% PSNR improvement | GitHub |
| Multi-Temporal | ResNet | SAR-optical fusion | Improved continuity | Research only |
| Cloud-GAN | Cyclic GAN | Unpaired training | High realism | Not available |

## 3. Shoreline Superresolution

The enhancement of shoreline features presents unique challenges due to the dynamic nature of coastal environments and the need to preserve temporal consistency. This section examines key approaches to superresolution specifically focused on coastal applications.

### Primary Methods

1. **CoastSat**
   - **Repository:** [CoastSat GitHub](https://github.com/kvos/CoastSat)
   - **Paper:** *CoastSat: A Google Earth Engine-Enabled Python Toolkit to Extract Shorelines from Publicly Available Satellite Imagery*
     **DOI:** [10.1016/j.envsoft.2019.104528](https://doi.org/10.1016/j.envsoft.2019.104528)
   - **Summary:** CoastSat is a Python toolkit leveraging Google Earth Engine to extract and monitor shoreline changes using Landsat and Sentinel imagery. The tool incorporates automated cloud masking and temporal smoothing to improve accuracy.
   - **Key Findings:** Accurate detection of shoreline positions with <2m RMSE in multiple test cases.
   - **Novel Contributions:** Simplifies large-scale shoreline monitoring with a user-friendly API.

2. **SRGAN for Remote Sensing**
   - **Paper:** *Super-Resolution of Sentinel-2 Images Using Deep Learning*
     **DOI:** [10.3390/rs12152207](https://doi.org/10.3390/rs12152207)
   - **Summary:** Adapts SRGAN to upscale Sentinel-2 imagery, improving resolution while preserving edge details like coastlines.
   - **Key Findings:** Achieves 2x spatial resolution enhancement with minimal spectral distortion.
   - **Novel Contributions:** First application of SRGAN for satellite-derived shoreline enhancement.

### Comparative Analysis

| Method | Key Features | Performance | Application Focus |
|--------|-------------|-------------|------------------|
| CoastSat | GEE integration, Automated processing | <2m RMSE | Operational monitoring |
| SRGAN-RS | Edge preservation, Spectral fidelity | 2x resolution | Research development |

## Implementation Resources

| Platform | Key Features | Use Case | Access |
|----------|-------------|----------|---------|
| Google Earth Engine | Large-scale processing, Data access | Production deployment | [Platform](https://earthengine.google.com/) |
| Microsoft Planetary Computer | Cloud computing, Data archives | Research development | [Documentation](https://planetarycomputer.microsoft.com/docs) |
