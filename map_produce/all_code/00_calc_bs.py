# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 00_calc_bs.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-23 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
import pandas as pd
import time
import numpy as np


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


gpd_pop = gpd.read_file('../../data/gadm36_shp/city-level-pop.geojson')
print(gpd_pop.head())

print('Loading BS data')
df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
# df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_1_percent.csv')
# df_bs = df_bs.sample(frac=0.01)

start = time.time()
print('Creating GeoDataFrame from bs data...')
gdf_bs = gpd.GeoDataFrame({'geometry': gpd.points_from_xy(df_bs['lon'],
                                                          df_bs['lat'],
                                                          crs='epsg:4326'),
                           'bs': 1.0, 'date': df_bs['created']})
# gdf_bs = GeoDataFrame({'geometry': PointArray(df_bs[['lon', 'lat']].values.tolist()), 'bs': 1.0, 'date': df_bs['created']})
print('Time: {:}'.format(time.time() - start))
# start = time.time()
# print('Writing GeoDataFrame to disk...')
# gdf_bs.to_geopandas().to_file('../../data/gadm36_shp/bs-01.shp')
# print('Time: {:}'.format(time.time() - start))

for year in range(2011, 2023):
    print(year)
    start = time.time()
    dates = pd.to_datetime(str(year))
    unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]

    geo_bs = gpd.sjoin(gpd_pop, bs_year)
    bs_grouped = geo_bs.groupby('NAME_2').agg({'bs': 'sum'}).reset_index()
    pop_bs = pd.merge(gpd_pop, bs_grouped, on='NAME_2', how='left')
    pop_bs['bs'].fillna(0, inplace=True)

    gpd_pop[str(year)] = imbalance_index(pop_bs['pop'], pop_bs['bs'])
    gpd_pop['bs_'+str(year)] = pop_bs['bs']
    print('Processing year {} took {:} seconds'.format(year, time.time() - start))

start = time.time()
print('Saving to disk')
gpd_pop.to_file('d:/all_bs.geojson', driver='GeoJSON')
print('Time: {:}'.format(time.time() - start))

