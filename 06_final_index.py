# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 06_final_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-21 (YYYY-MM-DD)
-----------------------------------------------
"""
import os
import pandas as pd
import geopandas as gpd
import mobile_codes
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
    imbalance_value = 0
    # we add a small term on the denominator to avoid zero divided
    revised_bs = users_per_bs * bs
    fill_values = np.full(1, fill_value=np.inf)
    p_b = np.divide(pop, revised_bs, out=fill_values, where=revised_bs != 0)
    log_p_b = np.log(p_b)
    a_exp_b = a * np.power(log_p_b, b)
    exp_a = 1. + np.exp(-a_exp_b)
    inverse = 1. / exp_a
    if pop > users_per_bs * bs:
        imbalance_value = 2. * inverse - 1.  # rescale
        imbalance_value = imbalance_value[0]
    return imbalance_value


def main():
    # worldwide boundaries in shape file
    folder = 'data/'
    save_folder = 'data/division/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    df_pop = pd.read_csv(save_folder + 'world_wide_pop.csv', sep=',')
    # BS file
    print('Loading BS data')
    df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    print('Loading country name data')
    df_country = pd.read_csv(save_folder + 'country_no_data.csv')

    results = []

    for name in df_country['region']:
        user_per_bs = 100
        if name == 'Syria':
            mcc_name = 'Syrian Arab Republic'
        elif name == 'Taiwan':
            mcc_name = 'Taiwan, Province of China'
        elif name == 'Russia':
            mcc_name = 'Russian Federation'
            user_per_bs = 40
        elif name == 'China':
            mcc_name = name
            user_per_bs = 1000
        else:
            mcc_name = name
        # if name != 'Syria':
        #     continue
        # mcc_name = 'Syrian Arab Republic'
        # print(mcc_name)
        try:
            mcc = np.array([mobile_codes.name(mcc_name).mcc], dtype=int).ravel()
            alpha3 = mobile_codes.name(mcc_name).alpha3
            pop = df_pop.loc[df_pop['region'] == name, 'pop'].values[0]
        except Exception as e:
            print('Something wrong on {:}'.format(name))
            continue
        bs = df_bs.loc[df_bs['mcc'].isin(mcc)].shape[0]

        # print(name, pop)
        index = imbalance_index(pop, bs, user_per_bs)
        results.append((alpha3, name, index))

    df_no_pop_index = pd.DataFrame(results, columns=['ISO Code', 'region', 'index'])
    df_no_pop_index.to_csv(save_folder+'country_no_data_index.csv', index=False)

    df_with_pop_index = pd.read_csv(save_folder+'worldwide_index.csv')
    all_df = pd.concat([df_no_pop_index, df_with_pop_index], ignore_index=True)
    name_list = [
        'Bolivia, Plurinational State of',
        'Congo, Democratic Republic of the',
        'Congo',
        'United Kingdom',
        'Iran, Islamic Republic of',
        'Lao People\'s Democratic Republic',
        'Moldova, Republic of',
        'Macedonia, the former Yugoslav Republic of',
        'Trinidad and Tobago',
        'Tanzania, United Republic of',
        'Saint Vincent and the Grenadines',
        'Venezuela, Bolivarian Republic of',
        'Viet Nam',
        'United States',
        'Korea, Republic of'
    ]
    correct_name = [
        'Bolivia', 'Democratic Republic of the Congo', 'Republic of Congo', 'UK',
        'Iran', 'Laos', 'Moldova', 'Macedonia', 'Trinidad', 'Tanzania',
        'Saint Vincent', 'Venezuela', 'Vietnam', 'USA', 'South Korea'
    ]
    for old, new in zip(name_list, correct_name):
        all_df.loc[all_df['region']==old, 'region'] = new
    # all_df.loc[all_df['region'].isin(name_list), 'region'] = correct_name
    all_df.to_csv(save_folder+'final_index.csv', index=False)


if __name__ == '__main__':
    main()
