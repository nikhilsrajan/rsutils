import numpy as np
import typing
import multiprocessing
import numba


def apply_cloud_mask(
    band_data:np.ndarray, 
    cloud_prob:np.ndarray, 
    cloud_prob_threshold:float=0.4,
    mask_value:typing.Any=np.nan,
)->np.ndarray:
    """
    Applies cloud mask for the given `cloud_prob_threshold` (0.4 by default)
    to the Nd array (N>=3) `band_data` as per the 3d array `cloud_prob` and return
    a Nd array (N>=3).

    Parameters
    ----------
    band_data : np.ndarray
        Expects a Nd array (N>=3): (n_timestamps, n_pixel_x, n_pixel_y, ...)
    cloud_prob : np.ndarray
        Excepts a 3d array: (n_timestamps, n_pixel_x, n_pixel_y)
    
    Returns
    -------
    band_masked_data : np.ndarray
        Returns a Nd array (N>=3): (n_timestamps, n_pixel_x, n_pixel_y, ...)
    """
    if (cloud_prob >= cloud_prob_threshold).sum() == 0:
        return band_data

    band_masked_data = band_data.copy()
    band_masked_data[np.where(cloud_prob >= cloud_prob_threshold)] = mask_value

    return band_masked_data


@numba.njit(parallel=True)
def _mask_interpolate_2D(band_data_2D:np.ndarray, mask_value:typing.Any=np.nan):
    n_samples, n_timestamps = band_data_2D.shape
    band_interp_data_2D = band_data_2D.copy()
    for n_sample in numba.prange(n_samples):
        if np.isnan(mask_value):
            valid_indexes = np.where(~np.isnan(band_data_2D[n_sample]))[0]
        else:
            valid_indexes = np.where(band_data_2D[n_sample] != mask_value)[0]
        n_valid_indexes = valid_indexes.shape[0]
        if n_valid_indexes == 0 or n_valid_indexes == n_timestamps:
            continue
        fp = np.array([band_data_2D[n_sample,valid_index] for valid_index in valid_indexes])
        band_interp_data_2D[n_sample] = np.interp(
            x=np.arange(n_timestamps),
            xp=valid_indexes,
            fp=fp
        )
    return band_interp_data_2D


def _flatten_2D(arr):
    """
    Flattens an N-D array to 2D array by keeping the 
    last dimension untouched.
    
    Input dimension: (n_1, n_2, ..., n_n, n_t)
    Output dimenstion: (n_1 * n_2 * ... * n_n, n_t)
    """
    *n_rem, n_ts = arr.shape
    return arr.reshape((np.prod(n_rem), n_ts))


def mask_interpolate(
    band_data:np.ndarray, 
    mask_value:typing.Any=np.nan,
    n_jobs=1,
):
    """
    Interpolates over nans across time-series.

    Parameters
    ----------
    band_data : np.ndarray
        Expects a N-d array - (n_0, n_1, n_2, ..., n_timestamps)
    
    Returns
    -------
    band_interp_data : np.ndarray
        Returns a N-d array - (n_0, n_1, n_2, ..., n_timestamps)
    """
    if np.isnan(mask_value):
        mask_count = np.isnan(band_data).sum()
    else:
        mask_count = (band_data == mask_value).sum()
        
    if mask_count == 0:
        return band_data

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    band_data_2D = _flatten_2D(band_data).astype(band_data.dtype)

    numba.set_num_threads(n=n_jobs)

    band_data_2D_interp = _mask_interpolate_2D(
        band_data_2D=band_data_2D,
        mask_value=mask_value,
    )

    return band_data_2D_interp.reshape(band_data.shape)


def apply_cloud_mask_and_interpolate(
    band_data:np.ndarray, 
    cloud_prob:np.ndarray, 
    cloud_prob_threshold:float=0.4,
    mask_value=np.nan,
    n_jobs=1,
)->np.ndarray:
    """
    Applies cloud mask for the given `cloud_prob_threshold` (0.4 by default)
    to the Nd array (N>=3) `band_data` as per the 3d array `cloud_prob`, interpolates
    over the masked values and return an Nd array (N>=3).

    Parameters
    ----------
    band_data : np.ndarray
        Expects a Nd array (N>=3): (n_timestamps, n_pixel_x, n_pixel_y, ...)
    cloud_prob : np.ndarray
        Excepts a 3d array: (n_timestamps, n_pixel_x, n_pixel_y)

    Returns
    -------
    band_interp_data : np.ndarray
        Returns a Nd array (N>=3): (n_timestamps, n_pixel_x, n_pixel_y, ...)
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    cloud_masked_band_data = apply_cloud_mask(
        band_data=band_data,
        cloud_prob=cloud_prob,
        cloud_prob_threshold=cloud_prob_threshold,
        mask_value=mask_value,
    )

    cloud_interp_band_data = mask_interpolate(
        band_data=cloud_masked_band_data.swapaxes(0,-1),
        mask_value=mask_value,
        n_jobs=n_jobs,
    ).swapaxes(0, -1)
    return cloud_interp_band_data


def angle_normalise(intensity_mat_in_db, angle_mat_in_deg):
   return intensity_mat_in_db + (angle_mat_in_deg - 38) * 0.13


def linear_to_db(intensity_mat_in_linear):
   return 10 * np.log10(intensity_mat_in_linear)


def db_to_linear(intensity_mat_in_db):
   return np.power(10, intensity_mat_in_db / 10)


def radar_scaling(
    band_data:np.ndarray,
    min_val:float=-50,
    max_val:float=1,
    linear_scale_input:bool=True,
    linear_scale_output:bool=True,
):
    """
    The current radar data preprocessing performed prior to training models
    """
    band_data_copy = band_data.copy()
    if linear_scale_input:
        band_data_copy = linear_to_db(intensity_mat_in_linear=band_data_copy)
    
    band_data_copy[np.where(band_data_copy>max_val)] = max_val
    band_data_copy[np.where(band_data_copy<min_val)] = min_val
    band_data_copy = (band_data_copy - min_val) / (max_val - min_val) # scale to [0,1]

    if linear_scale_output:
        band_data_copy = db_to_linear(intensity_mat_in_db=band_data_copy)
    
    return band_data_copy
