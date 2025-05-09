import rasterio
import rasterio.merge
import rasterio.mask
import rasterio.warp
import numpy as np
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import inspect
import glob
import shapely
import shapely.ops
import datetime
import gzip
import shutil    
import random
import string


def get_default_args(func):
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def driver_specific_meta_updates(meta, driver:str=None):
    if driver is None:
        driver = meta["driver"]
    if driver == "GTiff":
        meta.update({
            "driver": "GTiff",
            "compress": "lzw",
        })
    elif driver == "JP2OpenJPEG":
        # https://github.com/rasterio/rasterio/issues/1677#issuecomment-488597072
        meta.update({
            "driver": "JP2OpenJPEG",
            'QUALITY': '100',
            'REVERSIBLE': 'YES',
        })
    return meta


def keep_file_by_ext(
    filepath:str,
    ignore_extensions:list[str]=None,
    keep_extensions:list[str]=None,
):
    if ignore_extensions is None and keep_extensions is None:
        return True
    ignore_extension_present = False
    if ignore_extensions is not None:
        for ext in ignore_extensions:
            if filepath.endswith(ext):
                ignore_extension_present = True
                break
    keep_extension_present = False
    if keep_extensions is not None:
        for ext in keep_extensions:
            if filepath.endswith(ext):
                keep_extension_present = True
                break
    ret = (ignore_extensions is not None and not ignore_extension_present) or \
          (keep_extensions is not None and keep_extension_present)
    return ret


def get_all_files_in_folder(
    folderpath:str,
    ignore_extensions:list[str]=None,
    keep_extensions:list[str]=None,
):
    # https://stackoverflow.com/questions/18394147/how-to-do-a-recursive-sub-folder-search-and-return-files-in-a-list
    filepaths = [y for x in os.walk(folderpath) for y in glob.glob(os.path.join(x[0], '*')) if os.path.isfile(y)]
    
    filepaths = [fp for fp in filepaths if keep_file_by_ext(filepath=fp, 
                                                            ignore_extensions=ignore_extensions, 
                                                            keep_extensions=keep_extensions)]
    
    return filepaths


def modify_filepath(
    filepath:str, 
    prefix:str='', 
    suffix:str='', 
    new_folderpath:str=None, 
    new_ext:str=None,
    truncate_upto:int = None,
):
    folderpath, filename = os.path.split(filepath)
    if new_folderpath is not None:
        folderpath = new_folderpath
    filename_splits = filename.split('.')
    filename = filename_splits[0]
    
    # truncation is done since the filename gets absurdly large sometimes, too large to even create a file
    # if truncate_upto is None, no truncation happnes
    truncated_filename = filename[:truncate_upto] 

    ext = '.'.join(filename_splits[1:])
    if new_ext is not None:
        ext = new_ext
    return os.path.join(folderpath,f'{prefix}{truncated_filename}{suffix}.{ext}')


def create_fill_tif(
    reference_tif_filepath:str,
    out_tif_filepath:str,
    fill_value,
    nodata = None,
):
    with rasterio.open(reference_tif_filepath) as src:
        output_meta = src.meta.copy()
    if nodata is not None:
        output_meta['nodata'] = nodata
    full_ndarray = np.full(fill_value=fill_value, shape=(1, output_meta['height'], output_meta['width']), dtype=output_meta['dtype'])
    with rasterio.open(out_tif_filepath, 'w', **output_meta) as dst:
        dst.write(full_ndarray)


def create_zero_tif(
    reference_tif_filepath:str,
    zero_tif_filepath:str,
):
    create_fill_tif(
        reference_tif_filepath = reference_tif_filepath,
        out_tif_filepath = zero_tif_filepath,
        fill_value = 0,
        nodata = 1,
    )


def coregister(
    src_filepath:str,
    dst_filepath:str,
    reference_zero_filepath:str,
    resampling=rasterio.merge.Resampling.nearest,
    nodata=None,
):
    with rasterio.open(reference_zero_filepath) as ref:
        out_meta = ref.meta.copy()
    
    out_image, out_transform = rasterio.merge.merge(
        [reference_zero_filepath, src_filepath],
        method=rasterio.merge.copy_sum,
        resampling=resampling,
        nodata=nodata,
    )

    out_meta.update({
        'count': out_image.shape[0],
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform,
    })

    out_meta = driver_specific_meta_updates(meta=out_meta)

    with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
        dst.write(out_image)


def crop_tif(
    src_filepath:str, 
    shapes_gdf:gpd.GeoDataFrame,
    nodata = None,
    all_touched:bool = False,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(src_filepath) as src:
        out_meta = src.meta
        if nodata is None:
            nodata = out_meta['nodata']
        src_crs_shapes_gdf = shapes_gdf.to_crs(src.crs)
        shapes = src_crs_shapes_gdf['geometry'].to_list()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=nodata, all_touched=all_touched,
        )
        
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata,
    })

    out_meta = driver_specific_meta_updates(meta=out_meta)

    return out_image, out_meta


def resample_tif_inplace(
    src_filepath:str,
    ref_meta:dict,
    resampling = rasterio.warp.Resampling.average,
    src_nodata = None,
    dst_nodata = None,
    dst_dtype = None,
):
    """
    To warp one tif to match another's dimension
    """
    with rasterio.open(src_filepath) as src:
        src_meta = src.meta.copy()
        src_image = src.read()
        src_desc = src.descriptions

    if src_nodata is None:
        src_nodata = src_meta['nodata']

    if dst_nodata is None:
        ref_meta['nodata'] = src_nodata

    if dst_dtype is None:
        ref_meta['dtype'] = src_meta['dtype']
    else:
        ref_meta['dtype'] = dst_dtype

    ref_meta = driver_specific_meta_updates(meta=ref_meta, driver=src_meta['driver'])
    ref_meta['count'] = src_meta['count']

    dst_image = np.full(
        (ref_meta['count'], ref_meta['height'], ref_meta['width']), 
        dtype = ref_meta['dtype'],
        fill_value = ref_meta['nodata'] if ref_meta['nodata'] is not None else 0,
    )

    for i in range(ref_meta['count']):
        rasterio.warp.reproject(
            source = src_image[i],
            destination = dst_image[i],
            src_transform = src_meta['transform'],
            dst_transform = ref_meta['transform'],
            src_nodata = src_nodata,
            dst_nodata = ref_meta['nodata'],
            src_crs = src_meta['crs'],
            dst_crs = ref_meta['crs'],
            resampling = resampling,
        )
    
    return dst_image, src_desc


def resample_tif(
    ref_filepath:str,
    src_filepath:str,
    dst_filepath:str,
    resampling = rasterio.warp.Resampling.average,
    src_nodata = None,
    dst_nodata = None,
    dst_dtype = None,
):
    with rasterio.open(ref_filepath) as ref:
        ref_meta = ref.meta.copy()

    dst_image, src_desc \
    = resample_tif_inplace(
        src_filepath = src_filepath,
        ref_meta = ref_meta,
        resampling = resampling,
        src_nodata = src_nodata,
        dst_nodata = dst_nodata,
        dst_dtype = dst_dtype,
    )

    with rasterio.open(dst_filepath, 'w', **ref_meta) as dst:
        dst.descriptions = src_desc
        dst.write(dst_image)


def plot_clustered_lineplots(
    crop_name:str,
    band_name:str,
    timeseries:np.ndarray,
    x:list,
    cluster_ids:np.ndarray, 
    save_filepath:str,
    alpha:float=0.05,
    y_min:float=-0.3,
    y_max:float=0.9,
    scale:float=5,
    aspect_ratio:float=2,
    nrows:int=3,
    ncols:int=3,
    limit_plots_per_cluster:int=1000,
    random_state:int=42,
    x_label:str='dates',
    cluster_id_to_color_map:dict=None,
    x_label_rotation:float = 0,
):
    n_points, n_timestamps = timeseries.shape

    if len(x) != n_timestamps:
        raise ValueError('Length of y should match timeseries shape.')
    
    unique_cluster_ids, counts = np.unique(cluster_ids, return_counts=True)
    cluster_counts_df = pd.DataFrame(data={'cluster_id':unique_cluster_ids,'count':counts})
    sorted_cluster_ids = cluster_counts_df.sort_values(
        by=['count', 'cluster_id'], ascending=[False, True],
    )['cluster_id']

    n_clusters = len(unique_cluster_ids)
    if n_clusters > nrows * ncols:
        raise ValueError(
            f'Too many clusters for too less nrows and ncols. '
            f'n_clusters = {n_clusters} > nrows * ncols = {nrows * ncols}'
        )

    _df = pd.DataFrame(
        data=np.concatenate([
            np.array([range(n_points)]).T, 
            timeseries, 
            np.array([cluster_ids]).T
        ], axis=1),
        columns=['id'] + list(x) + ['cluster_id']
    )
    _df['id'] = _df['id'].astype(int)
    _df['cluster_id'] = _df['cluster_id'].astype(int)
    melted_dfs = []
    for cluster_id in sorted_cluster_ids:
        df_to_melt_i = _df[_df['cluster_id']==cluster_id]
        melted_df_i = df_to_melt_i[_df.columns[:-1]].melt(
            id_vars='id',
            var_name=x_label, 
            value_name=band_name,
        )
        melted_df_i['cluster_id'] = cluster_id
        melted_dfs.append(melted_df_i)
    _df_melted = pd.concat(melted_dfs)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(scale*aspect_ratio*ncols, scale*nrows))
    i, j = 0, 0

    for index, cluster_number in enumerate(sorted_cluster_ids):
        ax = axs[j][i]
        i += 1
        if i == ncols:
            j += 1
            i = 0
            
        _df_plot = _df_melted[_df_melted['cluster_id']==cluster_number]
        count = _df_plot['id'].unique().shape[0]
        ids = _df_plot['id'].unique().tolist()
        np.random.RandomState(seed=random_state).shuffle(ids)
        selected_ids = ids[:limit_plots_per_cluster]
        _df_plot = _df_plot[_df_plot['id'].isin(selected_ids)]
        limited_count = _df_plot['id'].unique().shape[0]

        if cluster_id_to_color_map is None:
            color = sns.palettes.color_palette()[index % 10]
        else:
            color = cluster_id_to_color_map[cluster_number]

        g = sns.lineplot(
            data=_df_plot,
            ax=ax,
            x=x_label,
            y=band_name,
            hue='id',
            alpha=alpha,
            legend=False,
            palette=[
                color for _ in range(limited_count)
            ],
        )
        g.set_ylim([y_min, y_max])
        g.set_title(f'crop: {crop_name}, cluster_id: {cluster_number}, count: {count}')
        g.grid()

        if x_label_rotation != 0:
            g.set_xticklabels(g.get_xticklabels(), rotation=x_label_rotation)

    while j < nrows:
        fig.delaxes(axs[j][i])
        i += 1
        if i == ncols:
            j += 1
            i = 0

    fig.savefig(save_filepath, bbox_inches='tight')

    plt.close()
    plt.cla()
    plt.clf()


def get_bounds_gdf(shapes_gdf:gpd.GeoDataFrame):
    bounding_polygon = shapely.convex_hull(shapely.ops.unary_union(shapes_gdf['geometry'])).envelope
    bounds_gdf = gpd.GeoDataFrame(data={'geometry': [bounding_polygon]}, crs=shapes_gdf.crs)
    return bounds_gdf


def get_bounds_gdf_from_image(
    filepath:str,
):
    with rasterio.open(filepath) as src:
        crs = src.crs
        bounds = src.bounds
    
    return gpd.GeoDataFrame(
        data = {
            'geometry': [
                shapely.Polygon([
                    [bounds.left, bounds.bottom],
                    [bounds.right, bounds.bottom],
                    [bounds.right, bounds.top],
                    [bounds.left, bounds.top],
                ])
            ]
        },
        crs = crs
    )


def get_actual_bounds_gdf(src_filepath:str, shapes_gdf:gpd.GeoDataFrame):
    bounds_gdf = get_bounds_gdf(shapes_gdf=shapes_gdf)
    out_image, out_meta = crop_tif(src_filepath=src_filepath, shapes_gdf=bounds_gdf)
    cropped_ndvi_tif_bounds = [
        out_meta['transform'] * (0, 0),
        out_meta['transform'] * (out_meta['width'], 0),
        out_meta['transform'] * (out_meta['width'], out_meta['height']),
        out_meta['transform'] * (0, out_meta['height']),
    ]
    actual_bounds_polygon = shapely.Polygon(cropped_ndvi_tif_bounds)
    actual_bounds_gdf = gpd.GeoDataFrame(
        data={'geometry': [actual_bounds_polygon]},
        crs=out_meta['crs'],
    )
    return actual_bounds_gdf


def decompress_gzip(gzip_filepath:str, out_filepath:str):
    with gzip.open(gzip_filepath) as gzip_file:
        with open(out_filepath, 'wb') as f_out:
            shutil.copyfileobj(gzip_file, f_out)


def read_tif(tif_filepath:str):
    with rasterio.open(tif_filepath) as src:
        ndarray = src.read()
        meta = src.meta.copy()
    return ndarray, meta


def get_random_alnum_str(length:int=5):
    return ''.join(random.choice(
        string.ascii_uppercase + string.ascii_lowercase + string.digits
    ) for _ in range(length))


def get_epochs_str(add_random_alnum:bool=True, length:int=5):
    random_alnum = ''
    if add_random_alnum:
        random_alnum = get_random_alnum_str(length=length)
    return f"{int(datetime.datetime.now().timestamp() * 1000000)}{random_alnum}"


def add_epochs_prefix(
    filepath, 
    prefix:str='', 
    new_folderpath=None, 
    add_random_alnum:bool=True, 
    length:int=5,
    truncate_upto:int=None,
):
    epoch_str = get_epochs_str(add_random_alnum=add_random_alnum, length=length)
    temp_prefix = f"{prefix}{epoch_str}_"
    temp_tif_filepath = modify_filepath(
        filepath = filepath,
        prefix = temp_prefix,
        new_folderpath = new_folderpath,
        truncate_upto = truncate_upto,
    )
    return temp_tif_filepath


class GZipTIF(object):
    def __init__(self, gzip_tif_filepath):
        self.gzip_tif_filepath = gzip_tif_filepath
        self.tif_filepath = None


    def _generate_temp_tif_filepath(self):
        gzip_tif_filepath_wo_ext = self.gzip_tif_filepath[:-3]
        temp_tif_filepath = add_epochs_prefix(
            filepath=gzip_tif_filepath_wo_ext, 
            prefix='temp_'
        )
        return temp_tif_filepath
    

    def decompress_and_load(self, tif_filepath:str=None):
        if self.tif_filepath is None:
            if tif_filepath is None:
                tif_filepath = self._generate_temp_tif_filepath()
            self.tif_filepath = tif_filepath
            decompress_gzip(
                gzip_filepath=self.gzip_tif_filepath, 
                out_filepath=self.tif_filepath,
            )
        return self.tif_filepath
    

    def delete_tif(self):
        if self.tif_filepath is not None:
            os.remove(self.tif_filepath)
            self.tif_filepath = None
    

    def __del__(self):
        self.delete_tif()
        

def get_mask_coords(
    mask_tif_filepaths:list[str],
):
    with rasterio.open(mask_tif_filepaths[0]) as src:
        meta = src.meta.copy()
    
    out_mask, out_transform = rasterio.merge.merge(
        mask_tif_filepaths,
        method=rasterio.merge.copy_max,
    )
    mask_xs, mask_ys = np.where(out_mask[0] == 1)

    crs = meta['crs']
    return mask_xs, mask_ys, out_transform, crs


def compute_longlat(x:float, y:float, shift:float, transform:rasterio.Affine)->tuple[float]:
    long, lat = transform * (y +  shift, x + shift)
    return long, lat


def add_point_geom_from_xy(
    row:dict,
    shift:float,
    transform:rasterio.Affine,
    x_col:str = 'x',
    y_col:str = 'y',
    point_geom_col:str = 'geometry',
):
    long, lat = compute_longlat(
        x = row[x_col],
        y = row[y_col],
        shift = shift,
        transform = transform,
    )
    row[point_geom_col] = shapely.Point(long, lat)
    return row


def create_xy_gdf(
    mask_tif_filepaths:list[str]
):
    mask_xs, mask_ys, mask_transform, mask_crs = \
    get_mask_coords(mask_tif_filepaths = mask_tif_filepaths)

    xy_gdf = gpd.GeoDataFrame(pd.DataFrame(
        data = {
            'x': mask_xs,
            'y': mask_ys,
        }
    ).apply(
        lambda row: add_point_geom_from_xy(
            row = row,
            shift = 0.5,
            transform = mask_transform,
            x_col = 'x',
            y_col = 'y',
            point_geom_col = 'geometry',
        ), axis=1
    ), crs=mask_crs)

    return xy_gdf

