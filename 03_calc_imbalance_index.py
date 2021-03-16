# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 03_calc_imbalance_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-14 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
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
    imbalance_values[pop <= users_per_bs*bs] = 0.0

    # we add a small term on the denominator to avoid zero divided
    revised_bs = users_per_bs*bs
    p_b = np.divide(pop, revised_bs, out=fill_values, where=revised_bs != 0)
    log_p_b = np.log(p_b)
    a_exp_b = a * np.power(log_p_b, b)
    exp_a = 1. + np.exp(-a_exp_b)
    inverse = 1. / exp_a
    imbalance_values[pop > users_per_bs*bs] = 2. * inverse - 1.  # rescale
    return imbalance_values


# load all the information
folder = 'D:/Dataset/HighResolutionPopulation/'
country = 'USA'
info_file = folder + country + '/all.geojson'
df_all = gpd.read_file(info_file)

print('Calculating imbalance index per grid...')

user_per_bs = 100
alpha = 1.0
beta = 1.0

df_all['imbalance_index'] = df_all['pop'] / (df_all['bs'] * user_per_bs)
df_all['new_imbalance_index'] = imbalance_index(df_all['pop'], df_all['bs'], user_per_bs, alpha, beta)
df_all.loc[(df_all['pop'] == 0) & (df_all['bs'] == 0), 'imbalance_index'] = 0.0

inf_imbalance = df_all.loc[df_all['imbalance_index'] == np.inf, 'pop'] / user_per_bs
df_all.loc[df_all['imbalance_index'] == np.inf, 'imbalance_index'] = inf_imbalance.values
df_all.to_file(folder+country+'/' + country + '_imbalance_index.geojson', driver='GeoJSON')

