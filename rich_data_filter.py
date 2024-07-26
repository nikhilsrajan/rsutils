import numpy as np
import numba

from . import utils_preprocess


@numba.njit()
def get_max_continuous_ones_1d(arr:np.ndarray)->np.ndarray:
    """
    Given a 1d array of 1s and 0s, this function returns the length of the longest
    sequence of 1s and the index from where the longest sequence starts.

    1d arr -> np.array([longest 1s sequence length, index of the longest 1s sequence])
    """
    n_timestamps = arr.shape[0]
    is_ones_sum = 0
    max_is_ones_sum = 0
    max_is_ones_start_index = 0
    prev = 0
    index = 0
    for i in numba.prange(n_timestamps):
        val = arr[i]
        if val == 1:
            is_ones_sum += 1
            if prev == 0:
                index = i
        elif prev != 0:
            if max_is_ones_sum < is_ones_sum:
                max_is_ones_sum = is_ones_sum
                max_is_ones_start_index = index
            is_ones_sum = 0
        prev = val
    if max_is_ones_sum < is_ones_sum:
        max_is_ones_sum = is_ones_sum
        max_is_ones_start_index = index
    return np.array([max_is_ones_sum, max_is_ones_start_index])


@numba.njit()
def get_max_continuous_ones_2d(arr:np.ndarray)->np.ndarray:
    n_samples, n_timestamps = arr.shape
    max_continuous_ones = np.zeros((n_samples, 2))
    for i in numba.prange(n_samples):
        max_continuous_ones[i] = \
            get_max_continuous_ones_1d(arr[i])
    return max_continuous_ones


def get_max_continuous_ones(arr:np.ndarray)->np.ndarray:
    org_shape = arr.shape
    arr_2D = utils_preprocess._flatten_2D(arr=arr)
    get_max_continuous_ones = get_max_continuous_ones_2d(arr=arr_2D)
    return get_max_continuous_ones.reshape(list(org_shape[:-1]) + [2])


def get_rich_data_stats(
    unavailable_data:np.ndarray,
):
    n_samples, n_timestamps, n_pixels_x, n_pixels_y, n_bands = unavailable_data.shape

    total_data_count = n_timestamps - unavailable_data.sum(axis=1)

    max_continuous_unavailable_data = get_max_continuous_ones(
        arr=unavailable_data.swapaxes(1,-1)
    ).swapaxes(1,-1)

    return total_data_count, max_continuous_unavailable_data


def get_rich_indices(
    unavailable_data:np.ndarray,
    min_total_data_count:int,
    max_max_continuous_unavailable:int,
    rich_data_min_proportion:float,
):
    total_data_count, max_continuous_unavailable_data \
    = get_rich_data_stats(unavailable_data=unavailable_data)

    max_continuous_unavailable = max_continuous_unavailable_data[:,0]

    data_availability_stats = (
        (total_data_count >= min_total_data_count) & \
        (max_continuous_unavailable <= max_max_continuous_unavailable)
    ).astype(int).mean(axis=(1,2,3))

    rich_data_indices = np.where(data_availability_stats>=rich_data_min_proportion)[0]

    return rich_data_indices
