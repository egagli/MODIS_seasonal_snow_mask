"""
Library of functions to create MODIS seasonal snow masks, snow appearance date, and snow disappearance date.

Author: Eric Gagliano (egagli@uw.edu)
Created: 04/2024
"""

import numpy as np
import xarray as xr

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

    effective_snow = xr.where((da.where(lambda x: x!=CLOUD_VALUE).ffill(dim='time') == SNOW_VALUE) & (da.where(lambda x: x!=CLOUD_VALUE).bfill(dim='time') == SNOW_VALUE),1,0).astype(bool)
    
    return effective_snow.chunk(dict(time=-1))

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
                if length > max_len:
                    max_len = length
                    max_start = current_start
                    max_end = i
                current_start = None
    if current_start is not None:
        length = len(arr) - current_start
        if length > max_len:
            max_len = length
            max_start = current_start
            max_end = len(arr) - 1

    if max_len == 0:
        return fill_value, fill_value, fill_value
    return max_start, max_end, max_len

def map_DOWY_values(value,substitution_dict):
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
    #return np.vectorize(substitution_dict.get)(value)
    return np.vectorize(lambda x: substitution_dict.get(x, fill_value), otypes=[np.int16])(value)
    

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
        get_longest_consec_stretch, 
        effective_snow_da,
        input_core_dims=[['time']],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.int16, np.int16, np.int16]
    )

    substitution_dict = {index: value for index, value in enumerate(effective_snow_da.DOWY.values)}


    snow_start_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[0],
        kwargs={'substitution_dict': substitution_dict},
        vectorize=True,
        dask='parallelized'
    )
    
    snow_end_DOWY = xr.apply_ufunc(
        map_DOWY_values,
        results[1],
        kwargs={'substitution_dict': substitution_dict},
        vectorize=True,
        dask='parallelized'
    )
    
    snow_mask = xr.Dataset({
        'SAD_DOWY': snow_start_DOWY,
        'SDD_DOWY': snow_end_DOWY,
        'max_consec_snow_days': snow_end_DOWY-snow_start_DOWY
    })

    for var in snow_mask:
        snow_mask[var].rio.write_nodata(fill_value, encoded=False, inplace=True)

    return snow_mask