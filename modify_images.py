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

from . import utils


"""
Logic here is that every function mandatorily takes in two inputs in the
order -- data and profile.
"""

def delete_aux_xml(jp2_filepath):
    aux_xml_filepath = jp2_filepath + '.aux.xml'
    if os.path.exists(aux_xml_filepath):
        os.remove(aux_xml_filepath)


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
        data, profile = utils.crop_tif(
            src_filepath, **first_kwargs
        )
        sequence = sequence[1:]
    else:
        with rasterio.open(src_filepath) as src:
            data = src.read()
            profile = src.meta.copy()

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


def crop(
    data:np.ndarray,
    profile:dict,
    shapes_gdf:gpd.GeoDataFrame,
    nodata = None,
    all_touched:bool = False,
):
    out_profile = profile.copy()
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


def resample_by_ref(
    data:np.ndarray,
    profile:dict,
    ref_filepath:str,
    resampling = rasterio.warp.Resampling.nearest, 
):
    with rasterio.open(ref_filepath) as ref:
        out_profile = ref.meta.copy()

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
