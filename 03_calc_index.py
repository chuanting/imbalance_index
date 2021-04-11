# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 03_calc_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-18 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import geopandas as gpd
import os
import numpy as np
import mobile_codes
import itertools


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
    folder = 'data/division/'
    final_results = []
    gsma_file = 'data/MCI_Data_2020.xls'
    gsma_data = pd.read_excel(gsma_file, skiprows=2, sheet_name=2)
    df_gsma = gsma_data.loc[gsma_data['Year'] == 2019]
    df_gsma = df_gsma[['Country', 'ISO Code', 'Index']]
    df_gsma.columns = ['Country', 'ISO Code', 'GSMA_index']
    inclusive_file = 'data/Inclusive_index.xls'
    inclusive_data = pd.read_excel(inclusive_file)
    inclusive_data.columns = ['Country', 'Inclusive_index']
    # find out countries in both index
    common_contry = pd.merge(left=inclusive_data, right=df_gsma, on='Country', how='left')
    common_contry.dropna(inplace=True)

    print(common_contry.head(10))
    for i, name in enumerate(common_contry['ISO Code']):
        print(i, name)
        whole_name = mobile_codes.alpha3(name).name
        gsma_index = common_contry.loc[common_contry['ISO Code'] == name, 'GSMA_index'].values[0]
        inclusive_index = common_contry.loc[common_contry['ISO Code'] == name, 'Inclusive_index'].values[0]
        if not os.path.exists(folder + name + '_all.geojson'):
            print('Not exists')
            continue
        df_all = gpd.read_file(folder + name + '_all.geojson')
        df_all = df_all.loc[df_all['pop'] > 0]
        print('Calculating imbalance index per grid...')
        new_values = []
        col_name = []

        for user_per_bs in [20, 40, 60, 80, 100]:
            for alpha in [0.2, 0.4, 0.6, 0.8, 1.0]:
                for beta in [0.2, 0.4, 0.6, 0.8, 1.0]:
                    # calculate the imbalance index
                    condition = 'rho_{:}_alpha_{:}_beta_{:}'.format(user_per_bs, alpha, beta)
                    col_name.append(condition)
                    df_all[condition] = imbalance_index(df_all['pop'], df_all['bs'], user_per_bs, alpha, beta)
                    new_index = df_all[condition].mean()
                    new_values.append(new_index)

        tmp = list(itertools.chain([name, whole_name, gsma_index, inclusive_index, *new_values]))
        final_results.append(tmp)
    header = ['ISO Code', 'Country', 'GSMA_Index', 'Inclusive_Index'] + col_name

    df = pd.DataFrame(final_results, columns=header)
    df.to_csv(folder + 'final_index.csv', index=False)


if __name__ == '__main__':
    main()
