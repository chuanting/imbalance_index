# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 04_world_wide_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-20 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import mobile_codes
import itertools
import glob


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
    folder = 'data/division/'
    final_results = []
    files = glob.glob(folder + '*.geojson')
    for i, file in enumerate(files):
        alpha3 = file[14:17]
        whole_name = mobile_codes.alpha3(alpha3).name
        # whole_name = mobile_codes.alpha3(file[10:13]).name
        print(i, whole_name)
        df_all = gpd.read_file(file)
        df_all = df_all.loc[df_all['pop'] > 0]
        print('Calculating imbalance index per grid...')

        user_per_bs = 100

        df_all['index'] = imbalance_index(df_all['pop'], df_all['bs'])
        new_index = df_all['index'].mean()
        final_results.append((alpha3, whole_name, new_index))
    header = ['ISO Code', 'region', 'index']

    df = pd.DataFrame(final_results, columns=header)
    df.to_csv(folder + 'worldwide_index.csv', index=False)


if __name__ == '__main__':
    main()
