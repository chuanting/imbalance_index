# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: average_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-16 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import geopandas as gpd
import os
import numpy as np


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


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
    folder = 'data/'

    final_results = []
    country_names = 'data/MCI_Data_2020.xls'
    df_country_info = pd.read_excel(country_names, skiprows=2, sheet_name=2)
    df_name = df_country_info.loc[df_country_info['Year'] == 2019]
    for i, name in enumerate(df_name['ISO Code']):
        print(i, name)
        GSMA_index = df_name.loc[df_name['ISO Code'] == name, 'Index'].values[0]
        if not os.path.exists(folder + name + '_index.geojson'):
            print('Not exists')
            continue
        # if name == 'USA':
        #     continue
        df_all = gpd.read_file(folder + name + '_index.geojson')
        df_all = df_all.loc[df_all['pop'] > 0]
        old_index = df_all['imbalance_index'].mean()
        # max_pop = df_all['pop'].max()
        # df_all.loc[df_all['bs'] == 0, 'new_imbalance_index'] = 1.0 - (df_all['pop'] / max_pop)
        new_index = (1.0 - df_all['new_imbalance_index'].mean()) * 100
        old_gini = gini_coefficient(df_all['imbalance_index'].values)
        new_gini = gini_coefficient(df_all['new_imbalance_index'].values)

        # calculate the imbalance index
        print('Calculating imbalance index per grid...')
        user_per_bs = 1
        alpha = 1.0
        beta = 1.0
        df_all['imbalance_index'] = df_all['pop'] / (df_all['bs'] * user_per_bs)
        df_all['new_imbalance_index'] = imbalance_index(df_all['pop'], df_all['bs'], user_per_bs, alpha, beta)
        df_all.loc[(df_all['pop'] == 0) & (df_all['bs'] == 0), 'imbalance_index'] = 0.0

        inf_imbalance = df_all.loc[df_all['imbalance_index'] == np.inf, 'pop'] / user_per_bs
        df_all.loc[df_all['imbalance_index'] == np.inf, 'imbalance_index'] = inf_imbalance.values

        print(name, old_index, new_index, old_gini, new_gini)
        final_results.append((name, GSMA_index, old_index, new_index, old_gini, new_gini))

    df = pd.DataFrame(final_results, columns=['country', 'GSMA', 'old_index', 'new_index', 'old_gini', 'new_gini'])
    df.to_csv(folder + 'division_level_index.csv', index=False)


if __name__ == '__main__':
    main()
