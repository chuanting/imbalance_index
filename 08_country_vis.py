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
import geopandas as gpd
import time


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
    division_folder = 'data/division/'
    grid_folder = 'data/grid/'
    # target_country = ['USA', 'SAU', 'BRA', 'UGA', 'TUN', 'TZA', 'ZAF', 'VNM', 'ROU', 'ESP', 'THA']
    target_country = ['FRA']
    # rho_values = [rho for rho in range(5, 105, 5)]
    rho_values = [100]
    col_names = [str(col) for col in rho_values]

    for i, name in enumerate(target_country):
        print(i, name)
        start = time.time()
        print('Now processing {:} data'.format(name))
        # get the number of BS and population per grid
        division_wise_info = gpd.read_file(division_folder + name + '_all.geojson')
        grid_wise_info = gpd.read_file(grid_folder + name + '_all.geojson')

        # calculate the imbalance index
        print('Calculating imbalance index per grid...')
        for j, upb in enumerate(rho_values):
            division_wise_info[col_names[j]] = imbalance_index(division_wise_info['pop'],
                                                               division_wise_info['bs'],
                                                               upb)
            grid_wise_info[col_names[j]] = imbalance_index(grid_wise_info['pop'],
                                                           grid_wise_info['bs'],
                                                           upb)
        division_wise_info = division_wise_info[['pop', 'bs', 'geometry'] + col_names]
        grid_wise_info = grid_wise_info[['pop', 'bs', 'geometry', 'lon', 'lat'] + col_names]

        print('Time used: {:.4f}'.format(time.time() - start))

        # output the obtained index results (division wise)
        division_wise_info.to_file(folder + name + '.division.shp')
        df_division_info = gpd.GeoDataFrame(division_wise_info)
        df_division_info.to_file(folder + name + '.division.geojson', driver='GeoJSON')

        # output the obtained index results (grid wise)
        grid_wise_info.to_file(folder + name + '.grid.shp')
        df_grid_info = gpd.GeoDataFrame(grid_wise_info)
        df_grid_info.to_file(folder + name + '.grid.geojson', driver='GeoJSON')


if __name__ == '__main__':
    main()
