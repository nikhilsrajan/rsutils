import rasterio
import rasterio.merge
import rasterio.mask
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


def get_default_args(func):
    # https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def keep_file_by_ext(
    filepath:str,
    ignore_extensions:list[str]=None,
    keep_extensions:list[str]=None,
):
    ignore_extension_present = False
    if ignore_extensions is not None:
        for ext in ignore_extensions:
            if filepath[-len(ext):] == ext:
                ignore_extension_present = True
                break
    keep_extension_present = False
    if keep_extensions is not None:
        for ext in keep_extensions:
            if filepath[-len(ext):] == ext:
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


def modify_filepath(filepath, prefix='', suffix='', new_folderpath:str=None):
    folderpath, filename = os.path.split(filepath)
    if new_folderpath is not None:
        folderpath = new_folderpath
    filename_splits = filename.split('.')
    filename = filename_splits[0]
    ext = '.'.join(filename_splits[1:])
    return os.path.join(folderpath,f'{prefix}{filename}{suffix}.{ext}')


def create_zero_tif(
    reference_tif_filepath:str,
    zero_tif_filepath:str,
):
    with rasterio.open(reference_tif_filepath) as src:
        output_meta = src.meta.copy()
    zero_ndarray = np.zeros(shape=(1, output_meta['height'], output_meta['width']), dtype=output_meta['dtype'])
    with rasterio.open(zero_tif_filepath, 'w', **output_meta) as dst:
        dst.write(zero_ndarray)


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
        'compress':'lzw',
    })

    with rasterio.open(dst_filepath, 'w', **out_meta) as dst:
        dst.write(out_image)


def crop_tif(src_filepath:str, shapes_gdf:gpd.GeoDataFrame):
    with rasterio.open(src_filepath) as src:
        out_meta = src.meta
        src_crs_shapes_gdf = shapes_gdf.to_crs(src.crs)
        shapes = src_crs_shapes_gdf['geometry'].to_list()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=out_meta['nodata']
        )
        
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "compress": "lzw",
    })

    return out_image, out_meta


def plot_clustered_lineplots(
    crop_name:str,
    band_name:str,
    timeseries:np.ndarray,
    y:list,
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
    y_label:str='dates',
    cluster_id_to_color_map:dict=None,
):
    n_points, n_timestamps = timeseries.shape

    if len(y) != n_timestamps:
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
        columns=['id'] + y + ['cluster_id']
    )
    _df['id'] = _df['id'].astype(int)
    _df['cluster_id'] = _df['cluster_id'].astype(int)
    melted_dfs = []
    for cluster_id in sorted_cluster_ids:
        df_to_melt_i = _df[_df['cluster_id']==cluster_id]
        melted_df_i = df_to_melt_i[_df.columns[:-1]].melt(
            id_vars='id',
            var_name=y_label, 
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
        np.random.RandomState(seed=random_state).shuffle(x=ids)
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
            x=y_label,
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
