# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 08_country_vis.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-22 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import mobile_codes
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray
import time
import os


def get_info_per_grid(bs, pop, poly):
    print('Creating GeoDataFrame from population data...')
    start = time.time()
    gdf_population = GeoDataFrame({'pop': pop['population'],
                                   'geometry': PointArray(pop[['longitude', 'latitude']].values.tolist())})
    print(gdf_population.shape)
    print('Time: {:}'.format(time.time() - start))

    start = time.time()
    print('Creating GeoDataFrame from bs data...')
    gdf_bs = GeoDataFrame({'geometry': PointArray(bs[['lon', 'lat']].values.tolist()),
                           'bs': 1.0})
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
    folder = 'data/target_country/'
    save_folder = 'data/division/'
    target_country = ['USA', 'SAU', 'FRA', 'UGA', 'TUN']
    for i, name in enumerate(target_country):
        print(i, name)
        print('Now processing {:} data'.format(name))
        # get the number of BS and population per grid
        all_info = gpd.read_file(save_folder + name + '_all.geojson')

        # calculate the imbalance index
        print('Calculating imbalance index per grid...')

        user_per_bs = 100
        if name == 'USA':
            user_per_bs = 50

        all_info['index'] = imbalance_index(all_info['pop'], all_info['bs'], user_per_bs)

        all_info = all_info[['UID', 'pop', 'GID_0', 'NAME_0', 'GID_1', 'NAME_1', 'GID_2', 'NAME_2', 'bs', 'index', 'geometry']]
        all_info.to_file(folder + name + '.shp')


if __name__ == '__main__':
    main()