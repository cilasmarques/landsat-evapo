# Landsat Evapotranspiration Calculator

A high-performance C++ implementation for calculating evapotranspiration from Landsat satellite imagery using Surface Energy Balance (SEB) models.

## Overview

This project implements the SEBAL (Surface Energy Balance Algorithm for Land) and STEEP (Surface Temperature and Energy Exchange Process) models using C++ for processing Landsat satellite data. The system processes multi-spectral Landsat imagery to estimate evapotranspiration, a crucial parameter for agricultural monitoring, water resource management, and climate studies.

## Features

- **High Performance**: Optimized C++ implementation for efficient computation
- **Multi-Sensor Support**: Compatible with Landsat 5, 7, and 8 satellite data
- **Multiple SEB Models**: Supports both SEBAL and STEEP methodologies
- **Comprehensive Outputs**: Generates evapotranspiration maps and related surface energy balance components
- **Performance Monitoring**: Built-in CPU and memory usage monitoring
- **Docker Integration**: Containerized environment for easy deployment
- **Automated Processing**: Streamlined workflow from data download to final results

## Architecture

The system processes Landsat imagery through several computational stages:

1. **Data Ingestion**: Reads multi-spectral bands and metadata
2. **Radiometric Correction**: Converts digital numbers to reflectance/radiance
3. **Vegetation Indices**: Calculates NDVI, SAVI, LAI, and PAI
4. **Surface Temperature**: Estimates land surface temperature from thermal bands
5. **Energy Balance**: Computes net radiation, soil heat flux, and sensible heat flux
6. **Endmember Selection**: Identifies hot and cold pixels for calibration
7. **Evapotranspiration**: Calculates final ET estimates using aerodynamic resistance

## Prerequisites

### System Requirements
- Linux-based system
- GCC compiler with C++14 support
- Docker (optional, for containerized execution)

### Dependencies
```bash
# Install system dependencies
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get install gdal-bin libtiff5-dev
```

## Installation

### Option 1: Native Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd landsat-evapo
   ```

2. **Build the C++ application**
   ```bash
   make build
   ```

### Option 2: Docker Installation

1. **Build the Docker image**
   ```bash
   make docker-build
   ```

2. **Test the Docker container**
   ```bash
   make docker-run
   ```

## Usage

### Data Preparation

1. **Download Landsat imagery**
   ```bash
   # Configure image parameters in Makefile
   IMAGE_LANDSAT="landsat_8"
   IMAGE_PATHROW="215065"
   IMAGE_DATE="2017-05-11"
   
   # Download data
   make docker-landsat-download
   ```

2. **Preprocess imagery**
   ```bash
   make docker-landsat-preprocess
   ```

3. **Fix permissions**
   ```bash
   make fix-permissions
   ```

### Execution

### Configuration Parameters

Edit the `Makefile` to customize execution parameters:

```makefile
# Processing method (0=SEBAL, 1=STEEP)
METHOD=0

# Input/Output paths
INPUT_DATA_PATH=./input/landsat_8_215065_2017-05-11/6502x7295
OUTPUT_DATA_PATH=./output
```

#### Native Execution

##### Landsat 8 Processing
```bash
# Standard execution
make exec-landsat8

# With performance analysis
make analisys-landsat8 # Multiple execution analysis
```

##### Landsat 5/7 Processing
```bash
# Standard execution
make exec-landsat5-7

# With performance analysis
make analisys-landsat5-7
```

#### Docker Execution

##### Landsat 8 Processing
```bash
# Standard execution
make docker-run-landsat8
```

##### Landsat 5/7 Processing
```bash
# Standard execution
make docker-run-landsat5-7
```

##### Interactive Docker Shell
```bash
# Run interactive shell in container
make docker-run
```

##### Clean Docker Resources
```bash
# Remove Docker image and containers
make docker-clean
```

## Input Data Structure

The system expects the following input files in the specified directory:

```
input/
└── landsat_8_215065_2017-05-11/
    └── 6502x7295/
        ├── B2.TIF      # Blue band
        ├── B3.TIF      # Green band
        ├── B4.TIF      # Red band
        ├── B5.TIF      # NIR band
        ├── B6.TIF      # SWIR1 band
        ├── B7.TIF      # SWIR2 band
        ├── B10.TIF     # Thermal band (L8)
        ├── elevation.tif # Digital elevation model
        ├── MTL.txt     # Landsat metadata
        └── station.csv # Weather station data
```

## Output Products

The system generates the following output files in the `output/` directory:

- **Evapotranspiration maps**: Daily and instantaneous ET estimates
- **Surface energy balance components**: Net radiation, sensible heat flux, latent heat flux
- **Vegetation indices**: NDVI, SAVI, LAI, PAI
- **Surface properties**: Albedo, surface temperature, emissivity
- **Performance metrics**: Execution time and resource utilization data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with different Landsat datasets
5. Submit a pull request

## Acknowledgments

- NASA/USGS for Landsat satellite data
- SEBAL and STEEP algorithm developers

