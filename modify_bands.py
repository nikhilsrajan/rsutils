import numpy as np
import numba
import tqdm
import scipy.signal

from . import utils
from . import utils_preprocess


def modify_bands_chunkwise(
    bands:np.ndarray,
    band_indices:dict,
    sequence:list,
    chunk_size:int = 5000,
    print_messages:bool = True
):
    list_new_bands = []
    for i in tqdm.tqdm(range(0,bands.shape[0],chunk_size)):
        _new_bands, _new_band_indices = modify_bands(
            bands = bands[i:i+chunk_size].copy(),
            band_indices = band_indices.copy(),
            sequence = sequence,
            print_messages = print_messages)
        list_new_bands.append(_new_bands)
    
    new_bands = np.concatenate(list_new_bands,axis=0)
    new_band_indices = _new_band_indices
    
    return new_bands,new_band_indices


def modify_bands(
    bands:np.ndarray,
    band_indices:dict,
    sequence:list,
    print_messages:bool=False
):
    for func, kwargs in sequence:
        if (print_messages):
            print(generate_preprocess_log_from_single_sequence(func = func,kwargs=kwargs))
        bands, band_indices = func(bands=bands, band_indices=band_indices, **kwargs)
    return bands, band_indices


def trim_bands(
    bands:np.ndarray,
    band_indices:dict,
    start_index:int,
    end_index:int,
):
    bands = bands[:,start_index:end_index]
    return bands, band_indices


def scale_bands(
    bands:np.ndarray,
    band_indices:dict,
    bands_to_scale:list[str],
    mean:float=0,
    std:float=1,
):
    band_indices_to_scale = [index for band, index in band_indices.items() if band in bands_to_scale]
    bands[:,:,:,:,band_indices_to_scale] = (bands[:,:,:,:,band_indices_to_scale] - mean) / std
    return bands, band_indices


def remove_bands(
    bands:np.ndarray,
    band_indices:dict,
    bands_to_remove:list[str],
):
    bands_to_keep = [band for band in band_indices.keys() if band not in bands_to_remove]
    band_indices_to_keep = [index for band, index in band_indices.items() if band in bands_to_keep]
    bands = bands[:,:,:,:,band_indices_to_keep]
    new_bands_indices = {band : new_index for new_index, band in enumerate(bands_to_keep)}
    return bands, new_bands_indices


def mask_invalid_and_interpolate(
    bands:np.ndarray,
    band_indices:dict,
    upper_cap:float=10000,
    lower_cap:float=0,
    mask_value:float=0,
    band_indices_to_modify:list=None,
):
    if band_indices_to_modify is None:
        band_indices_to_modify = band_indices.keys()

    selected_bands_indices = [index for band, index in band_indices.items() 
                               if band in band_indices_to_modify]
    
    original_dtype = bands.dtype
    if type(mask_value) == float and bands.dtype != float:
        bands = bands.astype(float)

    selected_bands = bands[:,:,:,:,selected_bands_indices]
    selected_bands[np.where(selected_bands >= upper_cap)] = mask_value
    selected_bands[np.where(selected_bands <= lower_cap)] = mask_value

    interp_selected_bands = utils_preprocess.mask_interpolate(
        band_data=selected_bands.swapaxes(1,-1), mask_value=mask_value,
    ).swapaxes(1,-1)
    interp_bands = bands
    interp_bands[:,:,:,:,selected_bands_indices] = interp_selected_bands

    interp_bands = interp_bands.astype(original_dtype)

    return interp_bands, band_indices


@numba.njit()
def get_mosaiced_n_timestamps(
    n_timestamps:int,
    step_size:int,
    window_size:int,
):
    """
    Assume a 10 length array:
    [0 1 2 3 4 5 6 7 8 9]

    We wish to mosaic with window size 4 and step size 6.
    With window size 4, we get the following 7 windows (groups):

    1 : [0 1 2 3]
    2 : [1 2 3 4]
    3 : [2 3 4 5]
    4 : [3 4 5 6]
    5 : [4 5 6 7]
    6 : [5 6 7 8]
    7 : [6 7 8 9]

    Number of windows can be calculated using: 
        length of array - window size + 1

    With step size 6 we select windows 1 and 7 (1+6)

    1 : [0 1 2 3]
    7 : [6 7 8 9]

    Number of windows selected using a given step size can be calculate using:
        ceil(number of windows / step size)

    Safest ceil(a / b) in python is -(a // -b) (Ref: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python)

    Therefore, number of windows selected = -(number of windows // - step size)
    """
    n_windows = n_timestamps - window_size + 1
    mosaiced_n_timestamps = -(n_windows // -step_size)
    return mosaiced_n_timestamps


@numba.njit()
def _median_mosaic_2D(
    bands_2D:np.ndarray,
    window_size:int,
    step_size:int,
):
    n_samples, n_timestamps = bands_2D.shape
    mosaiced_n_timestamps = get_mosaiced_n_timestamps(
        n_timestamps=n_timestamps,
        step_size=step_size,
        window_size=window_size
    )
    mosaiced_bands_2D = np.full(shape=(n_samples, mosaiced_n_timestamps), fill_value=np.nan)

    for i in numba.prange(n_samples):
        for t in range(mosaiced_n_timestamps):
            start_index = t*step_size
            end_index = start_index + window_size
            mosaic_group = bands_2D[i,start_index:end_index]
            mosaiced_bands_2D[i,t] = np.nanmedian(mosaic_group)

    return mosaiced_bands_2D


def median_mosaic(
    bands:np.ndarray,
    band_indices:dict,
    window_size:int,
    step_size:int,
    mask_value:float=np.nan,
):
    n_samples, n_timestamps, n_pixels_x, n_pixels_y, n_bands = bands.shape
    mosaiced_n_timestamps = get_mosaiced_n_timestamps(
        n_timestamps=n_timestamps, step_size=step_size, window_size=window_size,
    )
    
    dtype = bands.dtype
    bands = bands.astype(float)
    bands[np.where(bands==mask_value)] = np.nan

    bands_2D = bands.swapaxes(1, -1).reshape(n_samples*n_bands*n_pixels_x*n_pixels_y, n_timestamps)

    mosaiced_bands_2D = _median_mosaic_2D(
        bands_2D=bands_2D,
        window_size=window_size,
        step_size=step_size,
    )

    mosaiced_bands_2D = mosaiced_bands_2D.astype(dtype)
    mosaiced_bands = mosaiced_bands_2D.reshape(
        n_samples, n_bands, n_pixels_x, n_pixels_y, mosaiced_n_timestamps
    )
    mosaiced_bands = mosaiced_bands.swapaxes(1, -1)

    return mosaiced_bands, band_indices


def sav_gol(
    bands:np.ndarray,
    band_indices:dict,
    window_size:int=24,
    poly_order:int=2,
):
    dtype = bands.dtype
    bands = bands.astype(float)
    
    n_samples, n_timestamps, n_pixels_x, n_pixels_y, n_bands = bands.shape
    
    bands_2D = bands.swapaxes(1, -1).reshape(n_samples*n_bands*n_pixels_x*n_pixels_y, n_timestamps)
    
    bands_2D_filtered = scipy.signal.savgol_filter(bands_2D, window_size, poly_order)
    
    bands_2D_filtered = bands_2D_filtered.astype(dtype)
    
    bands_final = bands_2D_filtered.reshape(
        n_samples, n_bands, n_pixels_x, n_pixels_y,n_timestamps)
    bands_final = bands_final.swapaxes(1,-1)
    
    return bands_final, band_indices
    

def generate_preprocess_log_from_single_sequence(
    func,
    kwargs:dict
):
    arguments = utils.get_default_args(func=func)
    arguments.update(kwargs)
    for k in arguments.keys():
        v = arguments[k]
        if isinstance(v, float):
            if np.isnan(v):
                arguments[k] = 'nan'
    return {'function': func.__qualname__, 'kwargs': arguments}
    

def generate_preprocess_log_from_sequence(
    sequence:list,
):
    preprocess_log = []
    for func, kwargs in sequence: 
        preprocess_log.append(generate_preprocess_log_from_single_sequence(func = func,
                                                                           kwargs = kwargs,
                                                                           ))
    return preprocess_log


def generate_sequence_from_preprocess_log(
    preprocess_log:list
):
    name_to_function = {
        'trim_bands': trim_bands,
        'scale_bands': scale_bands,
        'remove_bands': remove_bands,
        'median_mosaic': median_mosaic,
        'sav_gol': sav_gol,
    }
    sequence = []
    for step in preprocess_log:
        func = name_to_function[step['function']]
        kwargs = step['kwargs']
        sequence.append((func, kwargs))
    return sequence

