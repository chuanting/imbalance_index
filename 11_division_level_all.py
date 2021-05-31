# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 02_division_level_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-18 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import mobile_codes
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray, GeometryArray, PolygonArray, MultiPolygonArray
import time
import os
from shapely.wkt import loads


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
                           'bs': 1.0, 'date': bs['created']})
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
    save_folder = 'data/final/division/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    poly_file = folder + 'gadm36_shp/world_all_levels.geojson'
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
        # print(i, name)
        # if i < 10:
        #     continue
        # if name != 'BEL':
        #     continue
        print('Now processing {:} data'.format(name))

        mcc = np.array([mobile_codes.alpha3(name).mcc], dtype=int).ravel()
        country_bs = df_bs.loc[df_bs['mcc'].isin(mcc)]
        country_pop_file = 'data/{:}.gz'.format(name)

        # load population data
        if not os.path.exists(country_pop_file):
            print('File not exists')
            continue

        df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
        # if name == 'USA':
        #     df_pop = df_pop.sample(frac=0.2)
        df_poly = worldwide.loc[worldwide['GID_0'] == name]

        if df_poly.shape[0] == 0:
            continue

        country_poly = worldwide.loc[(worldwide['GID_0'] == 'WWW') & (worldwide['NAME_0'] == df_poly['NAME_0'].unique()[0])]
        # print(country_poly)
        # country_poly['GID_0'] = 'WWW'

        df_poly = GeoDataFrame(df_poly)
        country_poly = GeoDataFrame(country_poly)

        print('Creating GeoDataFrame from population data...')
        start = time.time()
        gdf_population = GeoDataFrame({'pop': df_pop['population'],
                                       'geometry': PointArray(df_pop[['longitude', 'latitude']].values.tolist())})
        print(gdf_population.shape)
        print('Time: {:}'.format(time.time() - start))

        start = time.time()
        print('Creating GeoDataFrame from bs data...')
        gdf_bs = GeoDataFrame({'geometry': PointArray(country_bs[['lon', 'lat']].values.tolist()),
                               'bs': 1.0, 'date': country_bs['created']})
        print('Time: {:}'.format(time.time() - start))

        print('Calculating population per grid')
        start = time.time()
        pop_per_grid = sjoin(gdf_population, df_poly)
        pop_per_grid_group = pop_per_grid.groupby(['NAME_2']).sum().reset_index()[['NAME_2', 'pop']]

        country_poly['pop'] = gdf_population['pop'].sum()
        pop_all_grid = np.array(gdf_population['pop'].sum(), ndmin=1)
        print('Time: {:}'.format(time.time() - start))

        print('Calculating # of BS per grid')
        start = time.time()
        results = []
        for year in range(2014, 2023):

            # cp = country_poly.copy(deep=True)
            print(year)
            dates = pd.to_datetime(str(year))
            unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]

            bs_all_grid = np.array(bs_year['bs'].sum(), ndmin=1)

            bs_per_grid = sjoin(bs_year, df_poly)
            bs_per_grid_group = bs_per_grid.groupby('NAME_2').sum().reset_index()[['NAME_2', 'bs']]

            pop_align = pd.merge(left=pop_per_grid_group, right=df_poly, on='NAME_2', how='right')
            bs_align = pd.merge(left=bs_per_grid_group, right=df_poly, on='NAME_2', how='right')

            pop_align['pop'].fillna(0, inplace=True)
            bs_align['bs'].fillna(0, inplace=True)
            pop_align['bs'] = bs_align['bs']

            pop_align['year'] = year
            pop_align['index'] = imbalance_index(pop_align['pop'], bs_align['bs'])

            # cp['year'] = year
            # cp['bs'] = bs_year['bs'].sum()
            # cp['index'] = imbalance_index(pop_all_grid, bs_all_grid)

            results.append(pop_align)
            # results.append(cp)
            # country_poly.to_geopandas().to_file('d:/dd.geojson', driver='GeoJSON')
        print('Time: {:}'.format(time.time() - start))

        final_index = pd.concat(results)
        final_index = GeoDataFrame(final_index).to_geopandas()
        final_index.to_file(save_folder + name + '.division.geojson', driver='GeoJSON')
        # final_index_spd = GeoDataFrame(final_index).to_geopandas()
        # final_index_spd.to_file(save_folder + name + '_all.geojson', driver='GeoJSON')


if __name__ == '__main__':
    main()