from s2 import s2
import shapely.geometry
import shapely.affinity
import pandas as pd
import geopandas as gpd
import s2cell
import numpy as np


# source: https://s2geometry.io/resources/s2cell_statistics
RES_TO_KM_RANGE = {
    0: (7842, 7842),
    1: (3921, 5004),
    2: (1825, 2489),
    3: (840, 1167),
    4: (432, 609),
    5: (210, 298),
    6: (108, 151),
    7: (54, 76),
    8: (27, 38),
    9: (14, 19),
    10: (7, 9),
    11: (3, 5),
    12: (1699 / 1000, 2),
    13: (850 / 1000, 1185 / 1000),
    14: (425 / 1000, 593 / 1000),
    15: (212 / 1000, 296 / 1000),
}


def grid_size_2_res_mapping(grid_size_km:int=5):
    km_list= []
    res_list = []
    for res, km_range in RES_TO_KM_RANGE.items():
        km_list.append(km_range[0])
        km_list.append(km_range[1])
        res_list.append(res)
        res_list.append(res)
    
    km_list = np.array(km_list)
    km_diff = np.abs(km_list - grid_size_km)
    return res_list[np.argmin(km_diff)]
    

def get_s2_grids_gdf(
    geojson_epsg_4326:dict,
    grid_size_km:int=5,
    scale_fact:float=1.1,
    res:int=None,
)->gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame of s2 grids for a given geojson

    params:
    geojson_epsg_4326 : GeoJSON of the area of interest
    grid_size_km: Size of the grid in KM
    scale_fact: The factor to scale the grids from actual size,
                default is 1.1, means 10% extra boundary from each side.
    """
    if res is None:
        res = grid_size_2_res_mapping(grid_size_km=grid_size_km)
    shape = shapely.geometry.shape(geojson_epsg_4326)
    s2_grids = s2.polyfill(
        geo_json=shape.convex_hull.__geo_interface__,
        res=res,
        geo_json_conformant=True,
        with_id=True,
    )
    s2_grids_df = pd.DataFrame(data=s2_grids)
    s2_grids_df['geometry'] = s2_grids_df['geometry'].apply(shapely.geometry.Polygon)
    s2_grids_df = s2_grids_df[s2_grids_df['geometry'].apply(lambda x: shape.intersects(x))].reset_index(drop=True)
    s2_grids_df['geometry'] = s2_grids_df['geometry'].apply(
        lambda x: shapely.affinity.scale(x, xfact=scale_fact, yfact=scale_fact)
    )
    s2_grids_gdf = gpd.GeoDataFrame(s2_grids_df, crs='epsg:4326')
    return s2_grids_gdf


def get_grid_geometry_from_id(grid_id, scale_fact:float=1.1):
    """
    Returns a shapely polygon for a given grid_id
    """
    geometry = shapely.geometry.Polygon(s2.s2_to_geo_boundary(
        s2_address=grid_id, geo_json_conformant=True,
    ))
    geometry = shapely.affinity.scale(geometry, xfact=scale_fact, yfact=scale_fact)
    return geometry


def get_id_from_latlon(lat:float, lon:float, grid_size_km:float=5, res:int=None,):
    if res is None:
        res = grid_size_2_res_mapping(grid_size_km=grid_size_km)

    grid_id = s2cell.lat_lon_to_token(
        lat = lat,
        lon = lon,
        level = res,
    )

    return grid_id


def get_latlon_from_id(s2_grid_id:str):
    lat, lon = s2cell.token_to_lat_lon(
        token = s2_grid_id
    )
    return lat, lon


def get_grid_geometry_from_latlon(
    lat:float, 
    lon:float,
    grid_size_km:float=5,
    res:int = None,
    scale_fact:float=1.1,
):
    grid_id = get_id_from_latlon(
        lat = lat,
        lon = lon,
        grid_size_km = grid_size_km,
        res = res,
    )

    return get_grid_geometry_from_id(
        grid_id = grid_id,
        scale_fact = scale_fact,
    )
