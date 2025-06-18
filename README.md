# MODIS Seasonal Snow Cover Data

**Author:** Eric Gagliano  
**Date:** April 18th, 2024  
**Last updated:** June 6, 2025

## Overview

This repository processes MODIS MOD10A2 snow cover data from the Microsoft Planetary Computer to create seasonal snow presence products. For each water year, the pipeline generates maps showing:
- Maximum number of consecutive snow days for each pixel [max_consec_snow_days].
- Date of snow appearance / first day of snow cover for the max snow cover snow period [SAD_DOWY].
- Date of snow disappearance / first day of no snow cover for the max snow snow period [SDD_DOWY].

*Note: SAD_DOWY and SDD_DOWY dates are represented as day of water year, e.g. DOWY 1 in NH is October 1st.*

These outputs are useful for hydrologic, climate, and ecological studies that require spatially explicit seasonal snow persistence.

- [MODIS MOD10A2 Product Guide](https://nsidc.org/sites/default/files/mod10a2-v006-userguide_1.pdf)
- [Wrzesien et al. 2019 Data Product](https://zenodo.org/records/2626737)
- [MODIS Grid System](https://modis-land.gsfc.nasa.gov/MODLAND_grid.html)  

## Data product  

Check out the processed data product on zenodo for water years 2015-2024:

Gagliano, E. (2025). Global MODIS-derived seasonal snow cover (snow appearance date, disappearance date, and max consec snow days), water years 2015–2024 (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15692530


## Processing Steps

### 1. Create MODIS Grid and Tile List

- The script `make_grid_and_zarr_store.ipynb` constructs a global MODIS grid and determines which tiles overlap land areas.
- Using a land mask (from Natural Earth), MODIS tiles intersecting land are identified.
- A list of MODIS tiles to process is output to `modis_tile_processing_list.txt`.
- An empty cloud-friendly Zarr store is initialized for output data, with dimensions (water_year, y, x), and variables: `SAD_DOWY`, `SDD_DOWY`, `max_consec_snow_days`.

### 2. Download and Prepare MODIS Data

- For each tile in `modis_tile_processing_list.txt`, data is fetched from the Planetary Computer STAC API using `modis_masking.get_modis_MOD10A2_max_snow_extent`.
- The product is loaded as an xarray DataArray with the "Maximum Snow Extent" band.

### 3. Cloud and Darkness Handling

- MODIS data quality flags are inspected:
  - Cloud, darkness, and "no decision" pixels are all mapped to a special value (treated as cloud).
  - A cloud-filling approach is used: if snow is observed before and after a cloudy period, the period is assumed to be snow-covered (per Wrzesien et al. 2019).
- This step ensures snow duration is not underestimated due to clouds or polar night.

### 4. Binarize Snow Presence

- Data is binarized: 1 for snow presence (with cloud-filling), 0 for no snow.
- This is implemented by the function `binarize_with_cloud_filling` in `modis_masking.py`.

### 5. Water Year Alignment

- The time series is split into water years:
  - **Northern Hemisphere:** Water year starts October 1.
  - **Southern Hemisphere:** Water year starts April 1.
- The function `align_wy_start` ensures each water year starts with a valid (non-cloud) observation.

### 6. Calculate Snow Metrics

- For each pixel and water year, the following metrics are calculated:
  - **SAD (Snow Appearance Date):** First day of the longest continuous snowy period with snow.
  - **SDD (Snow Disappearance Date):** First day without snow after the longest snow stretch.
  - **Max Consecutive Snow Days:** Length of the longest continuous snowy period.
- Efficient, vectorized functions (with Numba) are used for performance.
- The output is written to the Zarr store in cloud storage.

### 7. Parallel/Batch Tile Processing

- Tiles are processed in parallel using Dask and Coiled for distributed cloud computation.
- Batches of tiles are submitted to the cluster, results are written, and the client is restarted between batches for robustness.

### 8. Output and Data Storage

- Final results are stored in a Zarr dataset, indexed by water year and spatial coordinates.
- The Zarr store is cloud-optimized for analysis-ready access.

## File Descriptions

- **global_processing.ipynb**: Main pipeline for batch processing MODIS tiles, cloud computation setup, and data writing.
- **make_grid_and_zarr_store.ipynb**: Constructs MODIS grid, outputs tile list, and initializes the Zarr store.
- **modis_masking.py**: Library for MODIS data processing, masking, cloud filling, and snow metric calculation.
- **modis_tile_processing_list.txt**: List of MODIS tiles (h/v grid) overlapping land, to be processed.
- **evaluation/**: (Directory for evaluation scripts and scratch work; content not listed here.)

## References

- Wrzesien, M. L., Pavelsky, T. M., Durand, M. T., Dozier, J., & Lundquist, J. D. (2019). Characterizing biases in mountain snow accumulation from global data sets. Water Resources Research, 55, 9872–9891. [doi:10.1029/2019WR024908](https://doi.org/10.1029/2019WR024908)
- [MODIS MOD10A2 documentation and product guides](https://nsidc.org/sites/default/files/mod10a2-v006-userguide_1.pdf)

## Example Usage

To process all tiles:
1. Ensure tile list and Zarr store are created (`make_grid_and_zarr_store.ipynb`).
2. Run the main processing pipeline (`global_processing.ipynb`) to compute snow metrics for each tile.
3. Use the output Zarr for downstream analysis or visualization.

---

For any questions or contributions, contact Eric Gagliano (egagli@uw.edu). I used github co-pilot to partially generate this readme!