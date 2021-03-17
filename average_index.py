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
        df_all = gpd.read_file(folder + name + '_index.geojson')
        df_all = df_all.loc[df_all['pop'] > 0]
        old_index = df_all['imbalance_index'].mean()
        # max_pop = df_all['pop'].max()
        # df_all.loc[df_all['bs'] == 0, 'new_imbalance_index'] = 1.0 - (df_all['pop'] / max_pop)
        new_index = (1.0 - df_all['new_imbalance_index'].mean()) * 100
        old_gini = gini_coefficient(df_all['imbalance_index'].values)
        new_gini = gini_coefficient(df_all['new_imbalance_index'].values)
        print(name, old_index, new_index, old_gini, new_gini)
        final_results.append((name, GSMA_index, old_index, new_index, old_gini, new_gini))

    df = pd.DataFrame(final_results, columns=['country', 'GSMA', 'old_index', 'new_index', 'old_gini', 'new_gini'])
    df.to_csv(folder + 'final_gini.csv', index=False)


if __name__ == '__main__':
    main()
