<div align="center">
  <h1 style="font-size: 3em;">LITTORAL</h1>
  <h2>Cloud Imputation Module</h2>
</div>

---

## Overview & Research Goals

### Research Question
**How can we best amalgamate existing, public satellite imagery into a dataproduct suitable for shoreline evolution timeseries analysis?**

### Research Goals

| Goal | Description | Implementation Status |
|------|-------------|---------------------|
| **Goal 1** | Infer the state of pixels obscured by clouds from adjacent clouds (adjacent in space or time) | 🔄 In development |
| **Goal 2** | Infer the state of pixels obscured by clouds from alternative imagery products (comparing imagery from different sources) | 🔄 In development |
| **Goal 3** | Superresolve the shoreline pixels potentially using the timeseries to train, but preserving fine grain changes to the shoreline over time | 🔄 In development |

> **Note**: Detailed methodology, literature review, and dataset specifications can be found in the `docs/` directory. For production deployments, raw Sentinel-2 imagery should be accessed via the [littoral_s2download](https://github.com/Wzesk/Littoral_S2download/) module.

---

This module represents one component in a broader pipeline for shoreline evolution analysis. The following section outlines how this module integrates with the complete processing workflow.

### Pipeline Integration

| Stage | Status | Purpose | Module |
|-------|--------|---------|---------|
| 1. Data Download | ✅ Separate Module | S2 imagery acquisition | [littoral_s2download](https://github.com/Wzesk/Littoral_S2download/) |
| **2. Cloud Detection** | ✅ This Module | Identify obscured regions | `cloud_mask.py` |
| **3. Cloud Imputation** | ✅ This Module | Reconstruct obscured pixels | `cloud_impute.py` |
| **4. Superresolution** | 🔄 In Progress | Enhance shoreline details | `superresolution.py` (planned) |
| 5. Segmentation | 📋 Planned | Feature detection | Future Module |
| 6. Edge Detection | 📋 Planned | Boundary extraction | Future Module |
| 7. Refinement | 📋 Planned | Boundary optimization | Future Module |
| 8. Georeferencing | 📋 Planned | Coordinate mapping | Future Module |

> **Production Note**: For large-scale deployments, use [littoral_s2download](https://github.com/Wzesk/Littoral_S2download/) for accessing and storing Sentinel-2 imagery via Google Cloud Platform (GCP). This ensures efficient data management and scalability.

### Core Files

| File | Purpose | Details |
|------|---------|---------|
| `cloud_mask.py` | Cloud Detection | - RGB+NIR cloud detection<br>- OmniCloudMask integration<br>- Mask generation and validation |
| `cloud_impute.py` | Cloud Imputation | - Prithvi-EO-2.0 implementation<br>- Training pipeline<br>- Inference optimization |
| `superresolution.py` | Resolution Enhancement | - Shoreline detail preservation<br>- Custom model architecture<br>- (In development) |

## Repository Structure

```plaintext
littoral_cloud_impute/
├── src/                  # Source code
│   ├── cloud_mask.py     # Cloud detection engine
│   ├── cloud_impute.py   # Imputation using Prithvi-EO-2.0
│   ├── superresolution.py # Resolution enhancement (planned)
│   └── inference.py      # Pipeline integration (planned)
├── tests/                # Unit tests
├── data/                 # Data directory (for development only; see below)
│   ├── raw/              # Local input S2 imagery (use GCP for production)
│   ├── processed/        # Processing outputs
│   └── metadata/         # Scene information
├── docs/                 # Documentation
│   ├── references.md     # Literature & resources
│   ├── dataset_requirements.md # Data specifications
│   └── research_plan.md  # Methodology
├── notebooks/            # Development notebooks
└── results/              # Visualizations
```

## Installation

### Option 1: Hatch (Recommended)
```bash
# Install hatch if needed
pip install hatch

# Clone and setup
git clone https://github.com/yourusername/littoral_cloud_impute.git
cd littoral_cloud_impute

# Create and activate environment
hatch env create
hatch shell
```

### Option 2: Virtual Environment
```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage Examples

### 1. Cloud Detection

```python
from PIL import Image
from littoral.cloud_mask import pred_clouds_from_rgbnir

# Load multispectral imagery
# For development, use local files. For production, see littoral_s2download.
rgb_img = Image.open('data/raw/sentinel2/2023_01_01/rgb.tif')
nir_img = Image.open('data/raw/sentinel2/2023_01_01/nir.tif')

# Generate cloud mask
mask, usable = pred_clouds_from_rgbnir(rgb_img, nir_img)
print(f"Scene usability: {usable:.2%}")

# Save results
mask.save('data/processed/cloud_masks/2023_01_01_mask.tif')
```

### 2. Cloud Imputation

```python
from littoral.cloud_impute import CloudImputation

# Initialize imputation model
imputer = CloudImputation(path="models/prithvi_params")

# Train on custom dataset (optional)
results = imputer.train(
    epochs=100,
    imgsz=640,
    batch=8,
    mask_ratio=4,
    name='shoreline_prithvi'
)

# Run inference
predictions = imputer.predict(image)
```

### 3. Full Pipeline (Planned)

```python
from littoral.inference import process_scene

# Process entire scene
results = process_scene(
    input_path='data/raw/sentinel2/2023_01_01/',
    output_path='data/processed/shorelines/',
    cloud_threshold=0.3,
    enhance_resolution=True
)
```

## Development

### Testing
```bash
# Run all tests
pytest tests/

# Test specific module
pytest tests/test_cloud_mask.py
```

### Code Quality
```bash
# Run linter
ruff check src/

# Auto-fix issues
ruff check --fix src/
```

## Documentation

Comprehensive documentation in `docs/`:
- `research_plan.md`: Methodology and objectives
- `references.md`: Literature and resources
- `dataset_requirements.md`: Data specifications

> **See Also**: For data acquisition workflows and cloud storage, refer to the [littoral_s2download](https://github.com/Wzesk/Littoral_S2download/) repository.

## License

MIT License

## Authors

- Walter Zesk (walter@littor.al)
- Philip Ndikum (philip-ndikum@users.noreply.github.com)
