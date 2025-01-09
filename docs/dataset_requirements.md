# Dataset Requirements

## Input Data Specifications

### Satellite Imagery Requirements

1. **RGB+NIR Data**
   - Format: GeoTIFF
   - Resolution: 10m (Sentinel-2) or 30m (Landsat)
   - Bands Required:
     - Red (B4 for Sentinel-2, B4 for Landsat)
     - Green (B3 for Sentinel-2, B3 for Landsat)
     - Blue (B2 for Sentinel-2, B2 for Landsat)
     - NIR (B8 for Sentinel-2, B5 for Landsat)

2. **Quality Requirements**
   - Atmospheric correction: Level-2A products preferred
   - Cloud coverage: All ranges acceptable (will be handled by pipeline)
   - Scene validity: >60% valid pixels preferred

## Data Organization

### Directory Structure
```
data/
├── raw/                  # Original satellite data
│   ├── sentinel2/       # Sentinel-2 imagery
│   │   ├── YYYY_MM_DD/  # Date-based organization
│   │   └── ...
│   └── landsat/         # Landsat imagery
│       ├── YYYY_MM_DD/
│       └── ...
├── processed/           # Pipeline outputs
│   ├── cloud_masks/    # Generated cloud masks
│   ├── imputed/        # Cloud-free reconstructions
│   └── shorelines/     # Final shoreline products
└── metadata/           # Data descriptions and logs
    ├── scene_metadata.csv
    └── processing_logs/
```

### Metadata Schema

#### Scene Metadata (scene_metadata.csv)
```csv
scene_id,date,satellite,path,row,cloud_cover,quality_score
```

#### Processing Logs
- Format: JSON
- Fields:
  - Processing timestamp
  - Input scene ID
  - Processing steps applied
  - Quality metrics
  - Output file locations

## Quality Control

### Validation Requirements
1. **Cloud Masks**
   - Manual validation subset
   - Comparison with standard products (e.g., Sentinel-2 SCL)
   - Accuracy metrics calculation

2. **Imputed Results**
   - Validation against cloud-free scenes
   - Temporal consistency checks
   - Spectral signature preservation

3. **Shoreline Products**
   - Comparison with high-resolution reference data
   - Temporal stability analysis
   - Uncertainty quantification

## Data Access

### Required Credentials
- Copernicus Open Access Hub account (for Sentinel-2)
- USGS Earth Explorer account (for Landsat)

### Data Sources
1. **Primary Sources**
   - [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
   - [USGS Earth Explorer](https://earthexplorer.usgs.gov/)

2. **Alternative Sources**
   - [Google Earth Engine](https://earthengine.google.com/)
   - [AWS Earth on AWS](https://aws.amazon.com/earth/)
   - [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
