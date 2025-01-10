# Methodology

## Overview
Our approach consists of three main components, each addressing a specific aspect of the shoreline analysis challenge. The methods are designed to work both independently and as part of an integrated pipeline.

## 1. Cloud Detection Pipeline

### Data Preprocessing
- Input: RGB+NIR bands from Sentinel-2/Landsat imagery
- Resolution: 10m (Sentinel-2) or 30m (Landsat)
- Bands:
  - Red (B4 for Sentinel-2, B4 for Landsat)
  - Green (B3 for Sentinel-2, B3 for Landsat)
  - Blue (B2 for Sentinel-2, B2 for Landsat)
  - NIR (B8 for Sentinel-2, B5 for Landsat)

### Cloud Detection Implementation
```python
def pred_clouds_from_rgbnir(rgb_img, nir_img):
    """
    Predict cloud and cloud shadow masks from RGB and NIR images.
    Returns binary mask and usable pixel ratio.
    """
    # Convert images to arrays
    rgb_arr = np.array(rgb_img)
    nir_arr = np.array(nir_img)

    # Create array with red, green and NIR
    img_arr = np.dstack([rgb_arr[:,:,0], rgb_arr[:,:,1], nir_arr[:,:,0]])

    # Generate predictions
    pred_mask = predict_from_array(img_arr)

    # Post-process results
    pred_array = np.where(pred_mask == 0, 1, 0) + np.where(pred_mask == 2, 1, 0)
    pred_array = np.where(pred_array > 0, 1, 0)

    return mask, usable_pixels
```

### Validation Metrics
- Cloud Detection Accuracy
- Shadow Detection Accuracy
- False Positive Rate
- False Negative Rate
- Usable Pixel Ratio

## 2. Cloud Imputation Methods

### Data Requirements
- Primary Image:
  - Cloud-masked RGB+NIR imagery
  - Associated cloud mask
- Alternative Sources:
  - Temporal sequence (previous/next clear views)
  - SAR data (when available)
  - Lower resolution but clear imagery

### Imputation Architecture
Based on Prithvi-EO-2.0:
```python
class CloudImputation:
    def __init__(self, path="/prithvi_params"):
        self.path = Path(path)
        self.yml_path = self.path / "data.yml"
        self.weights_path = self.path / "best.pt"

    def train(self, epochs=100, imgsz=640, batch=8, mask_ratio=4):
        """Train imputation model with shoreline-specific data"""
        pass

    def predict(self, image, model=None):
        """Apply cloud imputation to masked image"""
        pass
```

### Quality Assessment
- Spectral Consistency
- Spatial Coherence
- Temporal Stability
- Edge Preservation (especially at shorelines)

## 3. Shoreline Superresolution

### Input Requirements
- Cloud-free imagery (original or imputed)
- Historical shoreline data (when available)
- Terrain/bathymetry data (optional)

### Processing Steps
1. Region of Interest Detection
   - Identify water-land boundaries
   - Define buffer zones around shorelines

2. Resolution Enhancement
   - Apply specialized SR models to shoreline regions
   - Maintain temporal consistency
   - Preserve edge characteristics

3. Post-processing
   - Edge refinement
   - Noise reduction
   - Temporal smoothing

### Validation Approach
- Comparison with high-resolution reference data
- Temporal consistency metrics
- Edge preservation metrics
- Visual assessment by experts

## Integration Pipeline

### Workflow
1. Initial Assessment
   - Scene quality evaluation
   - Cloud coverage calculation
   - Data availability check

2. Processing Chain
   ```mermaid
   graph TD
   A[Input Image] --> B[Cloud Detection]
   B --> C[Cloud Mask]
   C --> D[Imputation]
   D --> E[Superresolution]
   E --> F[Final Output]
   ```

3. Quality Control
   - Automated metrics
   - Manual validation points
   - Temporal consistency checks

### Implementation Details
- Python-based pipeline
- GPU acceleration where available
- Modular design for component updates
- Extensive logging and validation

## Computational Requirements

### Hardware Recommendations
- GPU: NVIDIA RTX 3080 or better
- RAM: 32GB minimum
- Storage: SSD for processing

### Software Dependencies
- Python 3.8+
- PyTorch 2.0+
- GDAL for geospatial operations
- Custom libraries (detailed in requirements.txt)
