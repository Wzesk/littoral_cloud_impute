# Littoral Cloud Imputation

## Overview

The `littoral_cloud_impute` module provides intelligent cloud removal and inpainting for satellite imagery in the Littoral shoreline analysis pipeline. Using advanced deep learning techniques, this module reconstructs cloud-covered areas to produce clean, analysis-ready imagery essential for accurate shoreline detection.

## Pipeline Integration

This module serves as **Step 4** in the Littoral pipeline, processing coregistered satellite imagery after geometric alignment and before super-resolution upsampling. It implements a standardized interface that allows for easy integration and future model updates.

### Pipeline Context
```
Aligned Images → [Cloud Imputation] → Cloud-free Images → Upsampling → ...
```

### Interface
- **Input**: Coregistered satellite images with cloud contamination
- **Output**: Cloud-free imagery with reconstructed surface features
- **Function**: `vpint_cloud_impute.batch_remove_clouds_folder()`
- **Technology**: VPint2 (Value Propagation-based spatial interpolation) deep learning models

## Installation

### Environment Setup
```bash
conda create --name littoral_pipeline python=3.10
conda activate littoral_pipeline
# Install VPint dependencies
pip install -r requirements.txt
```

### Usage in Pipeline
```python
import sys
sys.path.append('/path/to/littoral_cloud_impute')
import vpint_cloud_impute

# Process folder of satellite images
folder_path = "/path/to/coregistered/images"
vpint_cloud_impute.batch_remove_clouds_folder(folder_path)

# Output will be saved in folder_path/cloudless/ with processing report
```

## Features

- **Advanced Inpainting**: Uses VPint2 deep learning for intelligent cloud reconstruction
- **Batch Processing**: Efficiently processes entire folders of satellite imagery  
- **Quality Assessment**: Generates detailed processing reports and success metrics
- **Multiple Formats**: Supports various satellite image formats (TIFF, PNG, etc.)
- **Integration Ready**: Standardized interface for seamless pipeline integration

## Output

The cloud imputation process produces:
- **Cloud-free Images**: Reconstructed imagery with clouds removed
- **Processing Reports**: Detailed CSV reports with success rates and quality metrics
- **Metadata Preservation**: Maintains geospatial information and image properties
- **Quality Filtering**: Automatic assessment of reconstruction quality

## Technology Stack

### VPint2 (Value Propagation-based Spatial Interpolation)
This module leverages the cutting-edge VPint2 algorithm for cloud removal, specifically designed for Earth observation data:

- **Training-free Approach**: No need for extensive training datasets
- **Thick Cloud Removal**: Capable of handling dense cloud coverage
- **Preserves Spatial Details**: Maintains fine-scale surface features during reconstruction
- **Robust Performance**: Works across diverse landscapes and acquisition conditions

## Credits and Attribution

This module is built upon the excellent **VPint** (Value Propagation-based spatial interpolation) framework:

### VPint2 Citation
When using this cloud imputation module, please cite the underlying VPint2 research:

```bibtex
@article{ArpEtAl24,
    author = {Laurens Arp and Holger Hoos and Peter {van Bodegom} and Alistair Francis and James Wheeler and Dean {van Laar} and Mitra Baratchi},
    title = {Training-free thick cloud removal for Sentinel-2 imagery using value propagation interpolation},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {216},
    pages = {168-184},
    year = {2024},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2024.07.030},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271624002995},
}
```

### VPint Original Citation
```bibtex
@article{ArpEtAl22,
    author = "Arp, Laurens and Baratchi, Mitra and Hoos, Holger",
    title = "VPint: value propagation-based spatial interpolation",
    journal = "Data Mining and Knowledge Discovery",
    volume = "36",
    publisher = "Springer",
    year = "2022",
    issn = "1573-756X",
    doi = "https://doi.org/10.1007/s10618-022-00843-2",
    url = "https://link.springer.com/article/10.1007/s10618-022-00843-2",
}
```

## Contributors

This module and the larger Littoral project has had numerous contributors, including:

**Core Development**: Walter Zesk, Tishya Chhabra, Leandra Tejedor, Philip Ndikum

**Project Leadership**: Sarah Dole, Skylar Tibbits, Peter Stempel

**VPint Development**: Laurens Arp, Mitra Baratchi, Holger Hoos, Alistair Francis, James Wheeler, Dean van Laar, Peter van Bodegom

## Reference

This project draws extensive inspiration from the [CoastSat Project](https://github.com/kvos/CoastSat):

Vos K., Splinter K.D., Harley M.D., Simmons J.A., Turner I.L. (2019). CoastSat: a Google Earth Engine-enabled Python toolkit to extract shorelines from publicly available satellite imagery. Environmental Modelling and Software. 122, 104528. https://doi.org/10.1016/j.envsoft.2019.104528 (Open Access)