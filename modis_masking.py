"""
Library of functions to create MODIS seasonal snow masks, snow appearance date, and snow disappearance date.

Author: Eric Gagliano (egagli@uw.edu)
Created: 04/2024
"""

import numpy as np
import pandas as pd
import xarray as xr
import pystac_client
import planetary_computer
import odc.stac
#import numba


def get_modis_MOD10A2_max_snow_extent(
    vertical_tile, horizontal_tile, start_date, end_date, chunks={"time": -1, "x": 240, "y": 240}
):

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[f"modis-10A2-061"],
        datetime=(start_date, end_date),
        query={
            "modis:vertical-tile": {"eq": vertical_tile},
            "modis:horizontal-tile": {"eq": horizontal_tile},
        },
    )

    load_params = {
        "items": search.item_collection(),
        "bands": "Maximum_Snow_Extent",
        "chunks": chunks,
    }

    modis_snow = odc.stac.load(**load_params)["Maximum_Snow_Extent"]

    return modis_snow


def get_modis_MOD10A2_full_grid():
    start_date="2020-09-22"
    end_date="2020-09-22"

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[f"modis-10A2-061"],
        datetime=(start_date, end_date),
    )

    load_params = {
        "items": search.item_collection(),
        "bands": "Maximum_Snow_Extent",
        "chunks": {},
    }

    modis_grid = odc.stac.load(**load_params).to_dataarray(dim='Maximum_Snow_Extent') #["Maximum_Snow_Extent"]

    return modis_grid

fill_value = np.iinfo(np.int16).min


def binarize_with_cloud_filling(da):
    """
    Binarize the MODIS DataArray with cloud filling.

    This function implements a cloud filling approach similar to the one described in Wrzesien et al. 2019.
    It assumes that if two snow-covered MOD10A2 observations bracket one or more cloudy
    MOD10A2 observations, the cloudy period is likely snow covered, too. Therefore, only
    the first and last 8-day MOD10A2 observations need to be snow covered. If there is one
    snowy MOD10A2 observation, five cloudy MOD10A2 periods, and one snowy observation,
    the 56-day period is classified as snow covered.

    This function should be run on the entire time series (not per water year groupby group) for continuity between water years. For example, let's say Dec 25 snow, Jan 2nd clouds, Jan 10 snow. If we groupby water year first, Jan 2nd would not be correctly identified as snow.

    Parameters:
    da (xarray.DataArray): The input MODIS MOD10A2 8 day DataArray.

    Returns:
    xarray.DataArray: The binarized DataArray (0: no snow, 1: snow), where 1 can be either
    snow or cloud(s) bracketed by snow.
    """
    SNOW_VALUE = 200
    CLOUD_VALUE = 50
    NO_SNOW_VALUE = 25
    DARKNESS_VALUE = 11
    NO_DECISION_VALUE = 1
    FILL_VALUE = 255

    # optionally replace darkness values with cloud value
    da = da.where(da != DARKNESS_VALUE, CLOUD_VALUE)
    # optionally replace FILL_VALUE with cloud value
    da = da.where(da != FILL_VALUE, CLOUD_VALUE)
    # optionally replace no decision values with cloud value
    da = da.where(da != NO_DECISION_VALUE, CLOUD_VALUE)


    # Avoid repeated ffill and bfill operations
    ffilled = da.where(lambda x: x != CLOUD_VALUE).ffill(dim="time")
    bfilled = da.where(lambda x: x != CLOUD_VALUE).bfill(dim="time")
    
    # Compute effective snow only once
    effective_snow = xr.where((ffilled == SNOW_VALUE) & (bfilled == SNOW_VALUE), 1, 0).astype(bool)

    if da.rio.crs is not None:
        effective_snow = effective_snow.rio.write_crs(da.rio.crs)

    return effective_snow


def get_longest_consec_stretch(arr):
    """
    Finds the longest consecutive stretch of snow days in a given array.

    This function iterates over the input array and finds the longest stretch of
    consecutive days where the value is True (indicating snow). It returns the start
    and end indices (end+1) of this stretch, as well as its length.

    Parameters:
    arr (list or array-like): The input array. Each element should be a boolean
    indicating whether there is snow on that day (True) or not (False).

    Returns:
    tuple: A tuple containing three elements:
        - The start index of the longest consecutive stretch of snow days.
        - The end index (+1) of the longest consecutive stretch of snow days.
        - The length of the longest consecutive stretch of snow days.
    """
    max_len = 0
    max_start = 0
    max_end = 0
    current_start = None
    for i, val in enumerate(arr):
        if val:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length >= max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = None
    if current_start is not None:
        length = len(arr) - current_start
        if length > max_len: # purposefully changed from >= to > to avoid including the last day as a SAD (avoid case where SAD=365,SDD=366,max_consec=1)
            max_len = length
            max_start = current_start
            max_end = len(arr) # used to be max_end = len(arr) -1, changed to include the last day in the stretch

    if max_len == 0:
        return fill_value, fill_value, fill_value
    return max_start, max_end, max_len

from numba import jit

@jit(nopython=True)
def get_longest_consec_stretch_vectorized(arr):
    """
    Optimized version using numba for speed.
    """
    n = len(arr)
    if n == 0:
        return fill_value, fill_value, fill_value
    
    max_len = 0
    max_start = 0
    max_end = 0
    current_start = -1
    
    for i in range(n):
        if arr[i]:  # Snow day
            if current_start == -1:
                current_start = i
        else:  # No snow day
            if current_start != -1:
                length = i - current_start
                if length >= max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = -1
    
    # Handle case where snow period extends to end
    if current_start != -1:
        length = n - current_start
        if length > max_len:
            max_len = length
            max_start = current_start
            max_end = n
    
    if max_len == 0:
        return fill_value, fill_value, fill_value
    
    return max_start, max_end, max_len

def map_DOWY_values(value, substitution_dict):
    """
    Maps the input values based on a predefined substitution dictionary.

    This function uses the 'np.vectorize' function to apply the 'get' method of the
    'substitution_dict' dictionary to the input values. The 'get' method returns the
    value for each key in the dictionary. If a key is not found in the dictionary,
    it returns None.

    Parameters:
    value (array-like): The input values to be mapped.

    Returns:
    numpy.ndarray: An array with the mapped values.
    """
    # return np.vectorize(substitution_dict.get)(value)
    return np.vectorize(
        lambda x: substitution_dict.get(x, fill_value), otypes=[np.int16]
    )(value)

def align_wy_start(da,hemisphere='northern'):
    """This function should operate on da and duplicate the last observation of each previous water year and relabel it as the start of the new water year, and then sort everything"""

    # Get unique water years
    water_years = da.water_year.values

    values, counts = np.unique(water_years, return_counts=True)

    # only inclue water years that have at least three observations
    valid_water_years = values[counts >= 5]
    
    # Create a new DataArray to hold the modified data
    new_data = []
    
    for wy in np.unique(valid_water_years):
        # Get the last observation of the current water year
        try:
            last_obs = da.where(da.water_year==wy-1,drop=True).isel(time=-1)
        except IndexError:
            print(f"Warning: No last observation of water year {wy-1}. This will affect calculation of water year {wy}, as the earliest possible snow appearance date will be DOWY 7 or 8. Skipping.")
            continue
        
        # Create a new observation for the start of the next water year
        new_obs = last_obs.copy()

        if hemisphere == 'northern':
            first_date_of_water_year = pd.to_datetime(f"{wy-1}-10-01")
        if hemisphere == 'southern':
            first_date_of_water_year = pd.to_datetime(f"{wy}-04-01")
        
        new_obs['time'] = first_date_of_water_year  # Set to October 1st of the next water year
        new_obs['water_year'] = wy
        new_obs['DOWY'] = 1
        
        # Append the original and new observations
        new_data.append(da.where(da.water_year==wy,drop=True))
        new_data.append(new_obs)
    
    # Concatenate all observations into a single DataArray
    combined_da = xr.concat(new_data, dim='time')
    
    # Sort by time
    combined_da = combined_da.sortby('time')

    combined_da = combined_da.where(combined_da.water_year.isin(valid_water_years), drop=True)
    combined_da = combined_da.astype(np.int16)

    
    return combined_da


def get_max_consec_snow_days_SAD_SDD_one_WY(effective_snow_da):
    """
    Calculates the maximum consecutive snow days, snow appearance day (SAD), and snow disappearance day (SDD) per water year.

    This function applies the 'get_longest_consec_stretch' function along the time dimension of the input DataArray to count
    consecutive snow days. It then maps the start and end days of the longest stretch of snow days to the 'DOWY' (Day of Water Year)
    coordinate of the input DataArray. The function returns a Dataset with three variables: 'SAD_DOWY', 'SDD_DOWY', and
    'max_consec_snow_days'.

    Parameters:
    effective_snow_da (xarray.DataArray): The input DataArray with effective snow data.

    Returns:
    xarray.Dataset: A Dataset with the following variables:
        - 'SAD_DOWY': The snow appearance day (SAD) for each water year, represented as a DOWY.
        - 'SDD_DOWY': The snow disappearance day (SDD) for each water year, represented as a DOWY. We say SDD is the first day with NO snow.
        - 'max_consec_snow_days': The maximum number of consecutive snow days for each water year.
    """

    # Apply function along the time dimension using the effective snow data to count consecutive snow days
    results = xr.apply_ufunc(
        get_longest_consec_stretch_vectorized,
        effective_snow_da,
        input_core_dims=[["time"]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={'allow_rechunk':True},
        output_dtypes=[np.int16, np.int16, np.int16],
    )

    substitution_dict = {
        index: value for index, value in enumerate(effective_snow_da.DOWY.values)
    }

    # add entry to substitution_dict for if at end of water year. SDD should be set to 366 for non-leap years and 367 for leap years.

    if effective_snow_da.time.dt.is_leap_year.any():
        last_dowy = 367
    else:
        last_dowy = 366

    substitution_dict[len(effective_snow_da.DOWY)] = last_dowy

    snow_start_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[0],
        kwargs={"substitution_dict": substitution_dict},
        vectorize=True,
        dask="parallelized",
    )

    snow_end_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[1],
        kwargs={"substitution_dict": substitution_dict},
        vectorize=True,
        dask="parallelized",
    )

    # if snow appearance date is last date

    snow_mask = xr.Dataset(
        {
            "SAD_DOWY": snow_start_DOWY,
            "SDD_DOWY": snow_end_DOWY,
            "max_consec_snow_days": snow_end_DOWY - snow_start_DOWY,
        }
    )

    snow_mask["max_consec_snow_days"] = snow_mask["max_consec_snow_days"].where(
        snow_mask["max_consec_snow_days"] > 0, fill_value
    )

    for var in snow_mask:
        snow_mask[var].rio.write_nodata(fill_value, encoded=False, inplace=True)

    return snow_mask


