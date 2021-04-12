# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: process_pop_bs.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-14 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import json
import time
import geopandas as gpd
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

if __name__ == '__main__':
    cluster = LocalCluster(dashboard_address=':8790',
                           n_workers=10,
                           threads_per_worker=1)
    client = Client(cluster)
    folder = 'D:/Dataset/HighResolutionPopulation/'
    country = 'USA'
    pop_file = folder + country + '/pop.csv'
    bs_file = folder + country + '/BS.csv'
    poly_file = folder + country + '/poly/poly.shp'
    grid_file = folder + country + '/grid.geojson'

    print('Loading population, bs, poly, and grid data...')
    df_pop = pd.read_csv(pop_file)
    df_bs = pd.read_csv(bs_file)
    gdf_poly = gpd.read_file(poly_file)
    gdf_poly = GeoDataFrame(gdf_poly)

    # gdf_poly.set_crs('epsg:4326', inplace=True, allow_override=True)
    start = time.time()
    grid = json.load(open(grid_file))
    print('Creating GeoDataFrame from grid data...')

    df_grid = gpd.GeoDataFrame.from_features(grid['features'])
    df_grid['id'] = range(df_grid.shape[0])

    df_bounds = df_grid['geometry'].bounds
    x = (df_bounds['minx'] + df_bounds['maxx']) / 2
    y = (df_bounds['miny'] + df_bounds['maxy']) / 2
    df_grid['lon'] = x.values
    df_grid['lat'] = y.values

    df_grid.set_crs('epsg:4326', inplace=True, allow_override=True)
    df_grid = GeoDataFrame(df_grid)

    df_country_geo = GeoDataFrame({'geometry': PointArray(df_grid[['lon', 'lat']].values.tolist()),
                                   'id': df_grid['id']})
    df_country_geo = dd.from_pandas(df_country_geo, npartitions=16).persist()
    df_country_grid = sjoin(df_country_geo, gdf_poly).compute()

    df_grid = df_grid.loc[df_country_grid.index]
    print('Time: {:}'.format(time.time() - start))

    print('Creating GeoDataFrame from population data...')
    start = time.time()
    gdf_population = GeoDataFrame({'pop': df_pop['Population'],
                                   'geometry': PointArray(df_pop[['Lon', 'Lat']].values.tolist())})
    # Large spatialpandas DaskGeoDataFrame with 16 partitions
    gdf_population = dd.from_pandas(gdf_population, npartitions=16).persist()
    # Pre-compute the partition-level spatial index
    gdf_population.partition_sindex
    print(gdf_population.shape)
    print('Time: {:}'.format(time.time() - start))

    start = time.time()

    print('Creating GeoDataFrame from bs data...')
    gdf_bs = GeoDataFrame({'geometry': PointArray(df_bs[['lon', 'lat']].values.tolist())})
    gdf_bs = dd.from_pandas(gdf_bs, npartitions=16).persist()
    gdf_bs.partition_sindex
    print('Time: {:}'.format(time.time() - start))

    print('Calculating population per grid')
    start = time.time()
    pop_per_grid = sjoin(gdf_population, df_grid).compute()
    pop_per_grid_group = pop_per_grid.groupby('id').sum().reset_index()[['id', 'pop']]
    print('Time: {:}'.format(time.time() - start))

    print('Calculating # of BS per grid')
    start = time.time()
    bs_per_grid = sjoin(gdf_bs, df_grid).compute()
    bs_per_grid_group = bs_per_grid.groupby('id').count().reset_index()[['id', 'index_right']]
    bs_per_grid_group.columns = ['id', 'bs']
    print('Time: {:}'.format(time.time() - start))

    print('Align BS and grid')
    start = time.time()
    # pop_align = pop_per_grid_group.merge(df_grid, )
    pop_align = pd.merge(left=pop_per_grid_group, right=df_grid, on='id', how='right')
    bs_align = pd.merge(left=bs_per_grid_group, right=df_grid, on='id', how='right')

    pop_align['pop'].fillna(0, inplace=True)
    bs_align['bs'].fillna(0, inplace=True)
    print('Time: {:}'.format(time.time() - start))

    pop_align['bs'] = bs_align['bs']

    print('Saving to local disk')
    start = time.time()
    all_info = GeoDataFrame(pop_align)
    all_info.to_geopandas().to_file(folder + country + '/all.geojson', driver='GeoJSON')
    print('Time: {:}'.format(time.time() - start))
