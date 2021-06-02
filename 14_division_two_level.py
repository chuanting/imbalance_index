# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 14_division_two_level.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-06-02 (YYYY-MM-DD)
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
    save_folder = 'data/final/division_two_levels/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    poly_file = folder + 'gadm36_shp/two-levels-4-digit.geojson'
    print('Loading worldwide boundary file')
    worldwide = gpd.read_file(poly_file)
    # BS file
    print('Loading BS data')
    df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    print('Loading country name data')
    country_names = worldwide['GID_0'].unique()
    for i, name in enumerate(country_names):
        print('Now processing {:} data'.format(name))
        try:
            codes = mobile_codes.alpha3(name)
            if codes.mcc is None:
                continue
        except Exception:
            print('Name {:} is not found'.format(name))
            continue

        mcc = np.array([codes.mcc], dtype=int).ravel()
        country_bs = df_bs.loc[df_bs['mcc'].isin(mcc)]

        country_pop_file = 'data/{:}.gz'.format(name)

        df_poly = worldwide.loc[worldwide['GID_0'] == name]
        if df_poly.shape[0] == 0:
            continue

        start = time.time()
        print('Creating GeoDataFrame from bs data...')
        gdf_bs = GeoDataFrame({'geometry': PointArray(country_bs[['lon', 'lat']].values.tolist()),
                               'bs': 1.0, 'date': country_bs['created']})
        print('Time: {:}'.format(time.time() - start))

        # load population data
        if not os.path.exists(country_pop_file):
            gdf_population = GeoDataFrame({'pop': 0, 'geometry': PointArray(df_poly.geometry.centroid)})
        else:
            df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
            gdf_population = GeoDataFrame({'pop': df_pop['population'],
                                           'geometry': PointArray(df_pop[['longitude', 'latitude']].values.tolist())})

        print('Calculating population per grid')
        start = time.time()
        df_poly = GeoDataFrame(df_poly)
        pop_per_grid = sjoin(gdf_population, df_poly)
        pop_per_grid_group = pop_per_grid.groupby(['NAME_1']).sum().reset_index()[['NAME_1', 'pop']]
        print('Time: {:}'.format(time.time() - start))

        print('Calculating # of BS per grid')
        start = time.time()
        results = []
        for year in range(2011, 2023):
            print(year)
            dates = pd.to_datetime(str(year))
            unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]

            bs_per_grid = sjoin(bs_year, df_poly)
            bs_per_grid_group = bs_per_grid.groupby('NAME_1').sum().reset_index()[['NAME_1', 'bs']]

            pop_align = pd.merge(left=pop_per_grid_group, right=df_poly, on='NAME_1', how='right')
            bs_align = pd.merge(left=bs_per_grid_group, right=df_poly, on='NAME_1', how='right')

            pop_align['pop'].fillna(0, inplace=True)
            bs_align['bs'].fillna(0, inplace=True)
            pop_align['bs'] = bs_align['bs']

            pop_align['year'] = year
            pop_align['index'] = imbalance_index(pop_align['pop'], bs_align['bs'])

            pop_align['pop'] = pop_align['pop'].round(4)
            pop_align['index'] = pop_align['index'].round(4)

            results.append(pop_align)
        print('Time: {:}'.format(time.time() - start))
        final_index = pd.concat(results)
        final_index = GeoDataFrame(final_index).to_geopandas()
        final_index.to_file(save_folder + name + '.division.geojson', driver='GeoJSON')


if __name__ == '__main__':
    main()