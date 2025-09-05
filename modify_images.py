import os
import rasterio
import rasterio.io
import rasterio.mask
import rasterio.warp
import geopandas as gpd
import multiprocessing as mp
import functools
import tqdm
import numpy as np
import rasterio.transform
import rasterio.merge

from . import utils


"""
Logic here is that every function mandatorily takes in two inputs in the
order -- data and profile.
"""

def delete_aux_xml(jp2_filepath):
    aux_xml_filepath = jp2_filepath + '.aux.xml'
    if os.path.exists(aux_xml_filepath):
        os.remove(aux_xml_filepath)


def modify_image_inplace(
    data:np.ndarray,
    profile:dict,
    sequence:list,
    raise_error:bool = True,
) -> tuple[np.ndarray, dict]:
    failed = False
    for func, kwargs in sequence:
        try:
            out_data, out_profile = func(
                data = data, 
                profile = profile, 
                **kwargs,
            )
            del data, profile
            data = out_data
            profile = out_profile
        except Exception as e:
            if raise_error:
                raise e
            else:
                failed = True
                break
    if failed:
        out_data = None
        out_profile = None
    
    return out_data, out_profile


def modify_image(
    src_filepath:str,
    dst_filepath:str,
    sequence:list,
    raise_error:bool = True,
):  
    """
    If the first function in the sequence is crop, this operation is performed
    from src_filepath. Performing it in memory is much slower since the whole image
    would be loaded. When cropping is performed from src_filepath, only the relevant
    part of the image is what is loaded thus saving time.
    """
    first_func, first_kwargs = sequence[0]
    if first_func == crop:
        try:
            data, profile = utils.crop_tif(
                src_filepath, **first_kwargs
            )
            sequence = sequence[1:]
        except Exception as e:
            if raise_error:
                raise e
            else:
                failed = True
    else:
        with rasterio.open(src_filepath) as src:
            data = src.read()
            profile = src.meta.copy()

    failed = False

    if not failed and len(sequence) > 0:
        out_data, out_profile = modify_image_inplace(
            data = data,
            profile = profile,
            sequence = sequence,
            raise_error = raise_error,
        )
        failed = out_data is None or out_profile is None

    if not failed:
        dst_folderpath = os.path.split(dst_filepath)[0]
        os.makedirs(dst_folderpath, exist_ok=True)
        with rasterio.open(dst_filepath, 'w', **profile) as dst:
            dst.write(data)
        delete_aux_xml(dst_filepath)

    return os.path.exists(dst_filepath) and not failed


def _modify_image_by_tuple(
    src_filepath_dst_filepath_tuple:tuple[str,str],
    sequence:list,
    raise_error:bool = True,
):
    src_filepath, dst_filepath = src_filepath_dst_filepath_tuple
    return modify_image(
        src_filepath = src_filepath,
        dst_filepath = dst_filepath,
        sequence = sequence,
        raise_error = raise_error,
    )


def modify_images(
    src_filepaths:list[str],
    dst_filepaths:list[str],
    sequence:list,
    njobs:int = mp.cpu_count() - 2,
    print_messages:bool = True,
    raise_error:bool = True,
):
    if len(src_filepaths) != len(dst_filepaths):
        raise ValueError('Size of src_filepaths and dst_filepaths do not match.')
    
    _modify_image_by_tuple_partial = functools.partial(
        _modify_image_by_tuple,
        sequence = sequence,
        raise_error = raise_error,
    )

    src_filepath_dst_filepath_tuples = list(zip(src_filepaths, dst_filepaths))

    with mp.Pool(njobs) as p:
        if print_messages:
            successes = list(tqdm.tqdm(
                p.imap(_modify_image_by_tuple_partial, src_filepath_dst_filepath_tuples), 
                total=len(src_filepath_dst_filepath_tuples)
            ))
        else:
            successes = list(p.imap(_modify_image_by_tuple_partial, src_filepath_dst_filepath_tuples))
    
    return successes


def _modify_image_inplace_by_tuple(
    data_profile:tuple[np.ndarray, dict],
    sequence:list,
    raise_error:bool = True,
) -> tuple[np.ndarray, dict]:
    data, profile = data_profile
    return modify_image_inplace(
        data = data,
        profile = profile,
        sequence = sequence,
        raise_error = raise_error,
    )


def modify_images_inplace(
    data_profile_list:list[tuple[np.ndarray, dict]],
    sequence:list,
    njobs:int = mp.cpu_count() - 2,
    print_messages:bool = True,
    raise_error:bool = True,
):
    _modify_image_inplace_by_tuple_partial = functools.partial(
        _modify_image_inplace_by_tuple,
        sequence = sequence,
        raise_error = raise_error,
    )

    with mp.Pool(njobs) as p:
        if print_messages:
            out_data_profile_list = list(tqdm.tqdm(
                p.imap(_modify_image_inplace_by_tuple_partial, data_profile_list), 
                total = len(data_profile_list)
            ))
        else:
            out_data_profile_list = list(p.imap(_modify_image_inplace_by_tuple_partial, data_profile_list))
    
    return out_data_profile_list


def load_image(
    src_filepath:str,
    shapes_gdf:gpd.GeoDataFrame = None,
    raise_error:bool = True,
    nodata = None,
    all_touched:bool = True,
) -> tuple[np.ndarray, dict]:
    
    data, profile = None, None

    try:
        if shapes_gdf is None:
            with rasterio.open(src_filepath) as src:
                data = src.read()
                profile = src.meta.copy()
        else:
            data, profile = utils.crop_tif(
                src_filepath = src_filepath, 
                shapes_gdf = shapes_gdf,
                nodata = nodata,
                all_touched = all_touched,
            )

    except Exception as e:
        if raise_error:
            raise e
        
    return data, profile


def load_images(
    src_filepaths:list[str],
    shapes_gdf:gpd.GeoDataFrame = None,
    raise_error:bool = True,
    nodata = None,
    all_touched:bool = True,
    njobs:int = 1,
    print_messages:bool = True,
):
    load_image_partial = functools.partial(
        load_image,
        shapes_gdf = shapes_gdf,
        raise_error = raise_error,
        nodata = nodata,
        all_touched = all_touched,
    )

    with mp.Pool(njobs) as p:
        if print_messages:
            data_profile_list = list(tqdm.tqdm(
                p.imap(load_image_partial, src_filepaths), 
                total = len(src_filepaths)
            ))
        else:
            data_profile_list = list(p.imap(load_image_partial, src_filepaths))
    
    return data_profile_list


def image_to_memfile(
    data:np.ndarray,
    profile:dict,
):
    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(data)
    return memfile


def images_to_memfiles(
    data_profile_list:list[tuple[np.ndarray, dict]],
):
    memfiles = []
    for data, profile in tqdm.tqdm(data_profile_list):
        memfiles.append(image_to_memfile(data=data, profile=profile))
    return memfiles


def crop(
    data:np.ndarray,
    profile:dict,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = None,
    all_touched:bool = False,
):
    out_profile = profile.copy()

    if nodata is None:
        nodata = profile['nodata']

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data)
        
            src_crs_shapes_gdf = shapes_gdf.to_crs(dataset.crs)
            shapes = src_crs_shapes_gdf['geometry'].to_list()

            out_data, out_transform = rasterio.mask.mask(
                dataset, shapes, crop=True, nodata=nodata, all_touched=all_touched,
            )

            out_profile.update({
                "height": out_data.shape[1],
                "width": out_data.shape[2],
                "transform": out_transform,
                "nodata": nodata,
            })
    
    return out_data, out_profile


def reproject(
    data:np.ndarray,
    profile:dict,
    dst_crs,
    resampling = rasterio.warp.Resampling.nearest,
):
    src_crs = profile['crs']
    src_count = profile['count']
    src_width = profile['width']
    src_height = profile['height']
    src_transform = profile['transform']

    # do nothing if crs is the same
    if src_crs == dst_crs:
        return data, profile

    src_bounds = rasterio.transform.array_bounds(height=src_height, width=src_width, transform=src_transform)

    transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds)
    
    out_profile = profile.copy()
    out_profile.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    out_profile = utils.driver_specific_meta_updates(meta=out_profile)

    out_data = np.full(
        (src_count, height, width), 
        dtype = profile['dtype'],
        fill_value = profile['nodata'],
    )

    for i in range(src_count):
        rasterio.warp.reproject(
            source = data[i],
            destination = out_data[i],
            src_transform = src_transform,
            src_crs = src_crs,
            dst_transform = transform,
            dst_crs = dst_crs,
            resampling = resampling)
    
    return out_data, out_profile


def resample_by_ref_meta(
    data:np.ndarray,
    profile:dict,
    ref_meta:dict,
    resampling = rasterio.warp.Resampling.nearest, 
):
    out_profile = ref_meta

    out_profile['nodata'] = profile['nodata']
    out_profile['dtype'] = profile['dtype']

    out_profile = utils.driver_specific_meta_updates(meta=out_profile, driver=profile['driver'])
    out_profile['count'] = profile['count']

    out_data = np.full(
        (out_profile['count'], out_profile['height'], out_profile['width']), 
        dtype = out_profile['dtype'],
        fill_value = out_profile['nodata'] if out_profile['nodata'] is not None else 0,
    )

    for i in range(out_profile['count']):
        rasterio.warp.reproject(
            source = data[i],
            destination = out_data[i],
            src_transform = profile['transform'],
            dst_transform = out_profile['transform'],
            src_nodata = profile['nodata'],
            dst_nodata = out_profile['nodata'],
            src_crs = profile['crs'],
            dst_crs = out_profile['crs'],
            resampling = resampling,
        )
    
    return out_data, out_profile


def resample_by_ref(
    data:np.ndarray,
    profile:dict,
    ref_filepath:str,
    resampling = rasterio.warp.Resampling.nearest, 
):
    with rasterio.open(ref_filepath) as ref:
        ref_meta = ref.meta.copy()

    return resample_by_ref_meta(
        data = data,
        profile = profile,
        ref_meta = ref_meta,
        resampling = resampling,
    )


def merge_inplace(
    data_profile_list:list[tuple[np.ndarray, dict]],
    nodata = None,
):
    merged_profile = data_profile_list[0][1].copy()

    memfiles = images_to_memfiles(
        data_profile_list = data_profile_list,
    )

    merged_data, merged_transform = rasterio.merge.merge(
        [memfile.open() for memfile in memfiles], 
        nodata = nodata,
    )

    merged_profile.update({
        'count': merged_data.shape[0],
        'height': merged_data.shape[1],
        'width': merged_data.shape[2],
        'transform': merged_transform,
    })

    return merged_data, merged_profile
