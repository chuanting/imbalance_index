# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: generate_usa.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-18 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import CRS
import geojson
import mobile_codes
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import time
import os
import dask


def get_info_per_grid(bs, pop, poly):
    print('Creating GeoDataFrame from population data...')
    start = time.time()
    gdf_population = GeoDataFrame({'pop': pop['population'],
                                   'geometry': PointArray(pop[['longitude', 'latitude']].values.tolist())})
    # Large spatialpandas DaskGeoDataFrame with 16 partitions
    # gdf_population = dd.from_pandas(gdf_population, npartitions=16).persist()
    # gdf_population = dd.from_pandas(gdf_population)
    # Pre-compute the partition-level spatial index
    # gdf_population.partition_sindex
    print(gdf_population.shape)
    print('Time: {:}'.format(time.time() - start))

    start = time.time()
    print('Creating GeoDataFrame from bs data...')
    gdf_bs = GeoDataFrame({'geometry': PointArray(bs[['lon', 'lat']].values.tolist()),
                           'bs': 1.0})
    # gdf_bs = dd.from_pandas(gdf_bs, npartitions=16).persist()
    # gdf_bs = dd.from_pandas(gdf_bs)
    # gdf_bs.partition_sindex
    print('Time: {:}'.format(time.time() - start))

    print('Calculating population per grid')
    start = time.time()
    pop_per_grid = sjoin(gdf_population, poly)
    pop_per_grid_group = pop_per_grid.groupby('UID').sum().reset_index()[['UID', 'pop']]
    print('Time: {:}'.format(time.time() - start))

    print('Calculating # of BS per grid')
    start = time.time()
    bs_per_grid = sjoin(gdf_bs, poly)
    bs_per_grid_group = bs_per_grid.groupby('UID').sum().reset_index()[['UID', 'bs']]
    # bs_per_grid_group.columns = ['UID', 'bs']
    print('Time: {:}'.format(time.time() - start))

    print('Align BS and grid')
    start = time.time()
    # pop_per_grid_group, bs_per_grid_group = dask.compute(pop_per_grid_group, bs_per_grid_group)
    pop_align = pd.merge(left=pop_per_grid_group, right=poly, on='UID', how='right')
    bs_align = pd.merge(left=bs_per_grid_group, right=poly, on='UID', how='right')

    pop_align['pop'].fillna(0, inplace=True)
    bs_align['bs'].fillna(0, inplace=True)
    print('Time: {:}'.format(time.time() - start))

    pop_align['bs'] = bs_align['bs']

    print('Saving to local disk')
    # print(pop_align)
    all_info = GeoDataFrame(pop_align)
    return all_info


def imbalance_index(pop, bs, users_per_bs=100, a=1.0, b=1.0):
    """
    :param pop: population of each grid
    :param bs: # of bs in each grid
    :param users_per_bs: how many users per bs can serve
    :param a: a parameter that controls the shape of the imbalance
    :param b: a parameter that controls the shape of the imbalance
    :return: imbalance index
    """
    imbalance_values = pop.copy()
    fill_values = np.full(shape=pop.shape[0], fill_value=np.inf)
    # when pop <= users_per_bs * bs, that is, this area is quite balance
    imbalance_values[pop <= users_per_bs * bs] = 0.0

    # we add a small term on the denominator to avoid zero divided
    revised_bs = users_per_bs * bs
    p_b = np.divide(pop, revised_bs, out=fill_values, where=revised_bs != 0)
    log_p_b = np.log(p_b)
    a_exp_b = a * np.power(log_p_b, b)
    exp_a = 1. + np.exp(-a_exp_b)
    inverse = 1. / exp_a
    imbalance_values[pop > users_per_bs * bs] = 2. * inverse - 1.  # rescale
    return imbalance_values


def main():
    # worldwide boundaries in shape file
    folder = 'data/'
    poly_file = folder + 'gadm36_shp/gadm36.shp'
    print('Loading worldwide boundary file')
    worldwide = gpd.read_file(poly_file)
    # BS file
    print('Loading BS data')
    df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    print('Loading country name data')
    country_names = 'data/MCI_Data_2020.xls'
    df_country_info = pd.read_excel(country_names, skiprows=2, sheet_name=2)
    df_name = df_country_info.loc[df_country_info['Year'] == 2019]
    for i, name in enumerate(df_name['ISO Code']):
        print(i, name)
        if name != 'USA':
            continue
        print('Now processing {:} data'.format(name))

        mcc = np.array([mobile_codes.alpha3(name).mcc], dtype=int).ravel()
        country_bs = df_bs.loc[df_bs['mcc'].isin(mcc)]
        country_pop_file = 'data/{:}.gz'.format(name)

        # load population data
        if not os.path.exists(country_pop_file):
            print('File not exists')
            continue

        df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
        if name == 'USA':
            df_pop = df_pop.sample(frac=0.2)
        df_poly = worldwide.loc[worldwide['GID_0'] == name]

        # # split the poly into many small grids
        # n_row, n_col = 200, 200
        # country_grid = split_poly_into_grids(df_poly, n_row, n_col)

        df_poly = GeoDataFrame(df_poly)
        # save the grid file
        # with open(folder + name + '_grid.geojson', 'w') as f:
        #     geojson.dump(country_grid, f)
        # df_grid = gpd.GeoDataFrame.from_features(country_grid['features'])

        # get the number of BS and population per grid
        start = time.time()
        all_info = get_info_per_grid(country_bs, df_pop, df_poly)
        all_info.to_geopandas().to_file(folder + name + '_all.geojson', driver='GeoJSON')
        print('Time: {:}'.format(time.time() - start))

        # calculate the imbalance index
        print('Calculating imbalance index per grid...')
        user_per_bs = 100
        alpha = 1.0
        beta = 1.0
        all_info['imbalance_index'] = all_info['pop'] / (all_info['bs'] * user_per_bs)
        all_info['new_imbalance_index'] = imbalance_index(all_info['pop'], all_info['bs'], user_per_bs, alpha, beta)
        all_info.loc[(all_info['pop'] == 0) & (all_info['bs'] == 0), 'imbalance_index'] = 0.0

        inf_imbalance = all_info.loc[all_info['imbalance_index'] == np.inf, 'pop'] / user_per_bs
        all_info.loc[all_info['imbalance_index'] == np.inf, 'imbalance_index'] = inf_imbalance.values
        all_info.to_geopandas().to_file(folder + name + '_index.geojson', driver='GeoJSON')


if __name__ == '__main__':
    main()