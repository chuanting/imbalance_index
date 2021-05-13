# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: process_geojson.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-03 (YYYY-MM-DD)
-----------------------------------------------
"""
import os
import glob
import geopandas as gpd
import pandas as pd


def main():
    folder_path = '../data/'
    bs_path = 'd:/dataset/cell_towers/cell_towers_2020-12-08-T000000.csv'
    country_list = ['AFG', 'AGO', 'AUS', 'CHN', 'USA']
    date_list = []
    # df_bs = pd.read_csv(bs_path)
    for slots in pd.date_range(start='1/1/2010', end='1/1/2020', freq='3M').astype(int) / 10**9:
        date_list.append(slots)
    all_country = []
    for country in country_list:
        country_pop_file = '../../data/{:}.gz'.format(country)
        country_geo_file = '../data/{}.geojson'.format(country)

        # # load population data
        # if not os.path.exists(country_pop_file):
        #     print('File not exists')
        #     continue
        # df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')

        # load boundary data
        df_geo = gpd.read_file(country_geo_file)

        df_geo['ISO3'] = country

        df_filter = df_geo.loc[df_geo['admin_level'].apply(lambda x: x in [2, 3, 4, 5, 6])]
        all_country.append(df_filter)

    df_all = gpd.GeoDataFrame(pd.concat(all_country))
    df_all.to_file('d:/worldwide.geojson', driver='GeoJSON')





if __name__ == '__main__':
    main()
