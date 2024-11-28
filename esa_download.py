import os
import requests
import shapely.geometry
import geopandas as gpd
import tqdm
import rasterio
import rasterio.mask
import rasterio.merge
import numpy as np
import pandas as pd
import multiprocessing as mp
import functools


DEFAULT_WORKING_DIR = 'esa'

VALID_YEARS = [2020, 2021]

TILE_ID_COL = 'll_tile'
YEAR_COL = 'year'
FILEPATH_COL = 'filepath'
EPSG_4326 = 'epsg:4326'

# source: https://esa-worldcover.org/en/data-access
ESA_S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
ESA_GRIDS_URL = f'{ESA_S3_PREFIX}/v100/2020/esa_worldcover_2020_grid.geojson'


ESA_LEGEND = {
    0: 'nodata',
    10: 'tree cover',
    20: 'shrubland',
    30: 'grassland',
    40: 'cropland',
    50: 'built-up', 
    60: 'Bare or sparse vegetation',
    70: 'snow and ice',
    80: 'permanent water bodies', 
    90: 'herbaceous wetland', 
    95: 'mangroves', 
    100: 'moss and lichen', 
}


def get_esa_tile_url(tile_id:str, year:int)->str:
    version = {
        2020: 'v100', 
        2021: 'v200', 
    }[year]
    return f'{ESA_S3_PREFIX}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile_id}_Map.tif'


def get_esa_grids_gdf(working_dir:str=DEFAULT_WORKING_DIR)->gpd.GeoDataFrame:
    esa_grid_filename = os.path.split(ESA_GRIDS_URL)[1]
    esa_grid_filepath = os.path.join(working_dir, esa_grid_filename)
    
    if os.path.exists(esa_grid_filepath):
        return gpd.read_file(esa_grid_filepath)
    
    os.makedirs(working_dir, exist_ok=True)
    esa_grids_gdf = gpd.read_file(ESA_GRIDS_URL)
    esa_grids_gdf.to_file(esa_grid_filepath)

    return esa_grids_gdf


def get_intersecting_tile_ids_single(
    geojson_epsg_4326:dict,
    working_dir:str=None,
)->list[str]:
    polygon = shapely.geometry.shape(geojson_epsg_4326)
    esa_grids_gdf = get_esa_grids_gdf(working_dir=working_dir)
    intersecting_tiles_gdf = esa_grids_gdf[esa_grids_gdf.intersects(polygon)]
    return intersecting_tiles_gdf[TILE_ID_COL].to_list()


def download_esa_tile(
    tile_id:str, 
    year:int,
    download_folderpath:str,
    overwrite:bool=False,
)->str:
    os.makedirs(download_folderpath, exist_ok=True)
    tile_url = get_esa_tile_url(tile_id=tile_id, year=year)
    tile_filename = os.path.split(tile_url)[1]
    download_filepath = os.path.join(
        download_folderpath,
        tile_filename,
    )
    if overwrite or not os.path.exists(download_filepath):
        response = requests.get(url=tile_url, allow_redirects=True)
        with open(download_filepath, 'wb') as fout:
            fout.write(response.content)
    
        # Compressing downloaded tif
        with rasterio.open(download_filepath) as src:
            imarray = src.read()
            output_meta = src.meta.copy()

        output_meta.update({"compress": "lzw"})

        with rasterio.open(download_filepath, 'w', **output_meta) as dst:
            dst.write(imarray)
    
    return download_filepath


def download_esa_tile_by_tuple(
    year_tile_id:tuple[int, str],
    download_folderpath:str,
    overwrite:bool=False,
):
    year, tile_id = year_tile_id
    return download_esa_tile(
        tile_id = tile_id,
        year = year,
        download_folderpath = download_folderpath,
        overwrite = overwrite,
    )


def download_esa_tiles(
    tile_ids:list[str],
    years:list[int],
    download_folderpath:str,
    overwrite:bool=False,
    njobs:int = mp.cpu_count() - 2,
):
    year_tile_id_list_of_tuples = [
        (year, tile_id) for year in years for tile_id in tile_ids
    ]

    download_esa_tile_by_tuple_partial = functools.partial(
        download_esa_tile_by_tuple,
        download_folderpath = download_folderpath,
        overwrite = overwrite,
    )

    with mp.Pool(njobs) as p:
        filepaths = list(tqdm.tqdm(
            p.imap(download_esa_tile_by_tuple_partial, year_tile_id_list_of_tuples), 
            total=len(year_tile_id_list_of_tuples)
        ))

    years, tile_ids = zip(*year_tile_id_list_of_tuples)
    
    filepaths_df = pd.DataFrame(data={
        YEAR_COL: years,
        TILE_ID_COL: tile_ids,
        FILEPATH_COL: filepaths,
    })

    return filepaths_df


def get_stats(tif_filepath:str, geojson_epsg_4326:dict):
    with rasterio.open(tif_filepath) as src:
        if src.crs != EPSG_4326:
            raise NotImplementedError('Written to handle only tifs of EPSG:4326 crs.')
        try:
            out_image, _ = rasterio.mask.mask(src, [geojson_epsg_4326], crop=True)
            labels, counts = np.unique(out_image, return_counts=True)
            return dict(zip(labels, counts))
        except ValueError:
            # print('No overlap between polygon and esa tile geotiff.')
            return {}
        

def esa_id_year_token(esa_id, year):
    return f'{esa_id} ({year})'
        

def get_esa_stats(
    geojson_epsg_4326:dict,
    tile_ids:list[str],
    years:list[int],
    tile_year_tif_filepath_dict:dict,
):
    agg_stats = {}
    for year in years:
        for tile_id in tile_ids:
            tif_filepath = tile_year_tif_filepath_dict[(tile_id, year)]
            stats = get_stats(
                tif_filepath=tif_filepath,
                geojson_epsg_4326=geojson_epsg_4326,
            )
            for k, v in stats.items():
                key = esa_id_year_token(esa_id=k, year=year)
                if k not in agg_stats.keys():
                    agg_stats[key] = 0
                agg_stats[key] += v

    return agg_stats


def get_esa_stats_by_tuple(
    id_geometry:tuple[str, shapely.Polygon],
    id_to_tile_ids_dict:dict,
    years:list[int],
    tile_year_tif_filepath_dict:dict,
    out_keys:list[str],
    id_col:str,
):
    _id, _geometry = id_geometry
    _agg_stats = get_esa_stats(
        geojson_epsg_4326 = _geometry.__geo_interface__,
        tile_ids = id_to_tile_ids_dict[_id],
        years = years,
        tile_year_tif_filepath_dict = tile_year_tif_filepath_dict,
    )

    out = {
        id_col: _id,
        'geometry': _geometry,
    }
    
    for key in out_keys:
        if key in [id_col, 'geometry']:
            continue
        elif key not in _agg_stats.keys():
            out[key] = 0
        else:
            out[key] = _agg_stats[key]
    
    return out


def get_intersecting_tile_ids(
    shapes_gdf:gpd.GeoDataFrame,
    id_col:str,
    working_dir:str = DEFAULT_WORKING_DIR,
):
    _shapes_gdf = shapes_gdf.to_crs(EPSG_4326)
    esa_grids_gdf = get_esa_grids_gdf(working_dir=working_dir)

    _sjoin_gdf =  gpd.sjoin(_shapes_gdf, esa_grids_gdf)

    tile_ids = _sjoin_gdf[TILE_ID_COL].unique()

    id_to_tile_ids_dict = _sjoin_gdf.groupby(id_col)[TILE_ID_COL].apply(list).to_dict()

    return tile_ids, id_to_tile_ids_dict


def download_intersecting_tiles(
    shapes_gdf:gpd.GeoDataFrame,
    id_col:str,
    years:list[int],
    overwrite:bool = False,
    working_dir:str = DEFAULT_WORKING_DIR,
    njobs:int = mp.cpu_count() - 2,
):
    invalid_years = list(set(years) - set(VALID_YEARS))

    if len(invalid_years) > 0:
        raise ValueError(f'Invalid years found: {invalid_years}')

    tile_ids, id_to_tile_ids_dict = \
    get_intersecting_tile_ids(
        shapes_gdf = shapes_gdf,
        id_col = id_col,
    )

    print('Downloading ESA tiles')
    filepaths_df = download_esa_tiles(
        tile_ids = tile_ids,
        years = years,
        download_folderpath = os.path.join(working_dir, 'tile'),
        overwrite = overwrite,
        njobs = njobs,
    )

    return id_to_tile_ids_dict, filepaths_df


def generate_esa_raster_stats_gdf(
    shapes_gdf:gpd.GeoDataFrame,
    id_col:str,
    years:list[int],
    overwrite:bool = False,
    working_dir:str = DEFAULT_WORKING_DIR,
    njobs:int = mp.cpu_count() - 2,
):
    id_to_tile_ids_dict, filepaths_df = \
    download_intersecting_tiles(
        shapes_gdf = shapes_gdf,
        id_col = id_col,
        years = years,
        overwrite = overwrite,
        working_dir = working_dir,
        njobs = njobs,
    )

    tile_year_tif_filepath_dict = dict(zip(
        zip(filepaths_df[TILE_ID_COL], filepaths_df[YEAR_COL]), 
        filepaths_df[FILEPATH_COL]
    ))

    out_keys = [
        id_col, 'geometry'
    ]

    for esa_id in ESA_LEGEND.keys():
        for year in years:
            out_keys.append(esa_id_year_token(esa_id=esa_id, year=year))

    _shapes_gdf = shapes_gdf.to_crs(EPSG_4326)

    id_geometry_list_of_tuples = list(zip(
        _shapes_gdf[id_col],
        _shapes_gdf['geometry'],
    ))

    print('Computing ESA stats')
    get_esa_stats_by_tuple_partial = functools.partial(
        get_esa_stats_by_tuple,
        id_to_tile_ids_dict = id_to_tile_ids_dict,
        years = years,
        tile_year_tif_filepath_dict = tile_year_tif_filepath_dict,
        out_keys = out_keys,
        id_col = id_col,
    )

    with mp.Pool(njobs) as p:
        out_list = list(tqdm.tqdm(
            p.imap(get_esa_stats_by_tuple_partial, id_geometry_list_of_tuples), 
            total=len(id_geometry_list_of_tuples)
        ))

    esa_stats_gdf = gpd.GeoDataFrame(
        pd.DataFrame.from_records(out_list),
        crs = EPSG_4326
    )

    return esa_stats_gdf
