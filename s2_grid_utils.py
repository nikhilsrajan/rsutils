from s2 import s2
import shapely.geometry
import shapely.affinity
import pandas as pd
import geopandas as gpd


def grid_size_2_res_mapping(grid_size_km:int=5):
    # source: https://s2geometry.io/resources/s2cell_statistics
    """
    based on grid_size provided by user - return resolution value
    :param grid_size:
    """
    if grid_size_km > 20 or grid_size_km < 1:
        raise NotImplementedError(
            f"grid size (km): {grid_size_km} not supported by our workflows at the moment"
            f"please provide anything b/w 1 to 20 both inclusive"
        )

    if 14 <= grid_size_km <= 20:
        return 9
    elif 7 <= grid_size_km <= 10:
        return 10
    elif 3 <= grid_size_km <= 5:
        return 11
    elif grid_size_km == 1:
        return 13
    else:
        raise NotImplementedError(
            f"grid size (km): {grid_size_km} not supported by our workflows at the moment"
            f"please provide anything from (1, 2, 5, 10)"
        )
    

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


def get_grid_geometry(grid_id, scale_fact:float=1.1):
    """
    Returns a shapely polygon for a given grid_id
    """
    geometry = shapely.geometry.Polygon(s2.s2_to_geo_boundary(
        s2_address=grid_id, geo_json_conformant=True,
    ))
    geometry = shapely.affinity.scale(geometry, xfact=scale_fact, yfact=scale_fact)
    return geometry
