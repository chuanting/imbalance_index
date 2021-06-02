# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 12_combine_all_countries_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-29 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import glob

# division_folder = 'data/final/division_three_levels/'
# division_results = []
# for i, file in enumerate(glob.glob(division_folder + '*.geojson')):
#     print(file)
#     country = gpd.read_file(file)
#     country['type'] = 'division'
#     country.to_file(file, driver='GeoJSON', index=False)
#     division_results.append(country)
#
# df_division = pd.concat(division_results, ignore_index=True)
# df_division['type'] = 'division'
#
# df_division.to_file('data/final/division_two_levels.geojson', driver='GeoJSON')

# for year in df_division['year'].unique():
#     df_year = df_division.loc[df_division['year'] == year]
#     df_year.to_file('data/final/division_year_{:}.geojson'.format(year), driver='GeoJSON')
#
# df_division = gpd.read_file('data/final/division.geojson')

# grid_folder = 'data/final/grid/'
# grid_results = []
# for i, file in enumerate(glob.glob(grid_folder + '*.geojson')):
#     print(file)
#     # if i > 5:
#     #     break
#     country = gpd.read_file(file)
#     country['type'] = 'grid'
#     country.to_file(file, driver='GeoJSON', index=False)
#     # grid_results.append(country)


# df_grid = pd.concat(grid_results, ignore_index=True)
# df_grid.drop(['lon', 'lat'], inplace=True, axis=1)
# df_grid['type'] = 'grid'
# df_grid.to_file('data/final/grid.geojson', driver='GeoJSON')

# final_index = pd.concat([df_division, df_grid], ignore_index=True)
# final_index.to_file('d:/final_index.geojson', driver='GeoJSON', index=False)


# country level index

division_folder = 'data/final/division_one_level/'
country = gpd.read_file('data/gadm36_shp/country-level-4-digit.geojson')
division_results = []
for i, file in enumerate(glob.glob(division_folder + '*.geojson')):
    print(file)
    df = gpd.read_file(file)

    x = pd.merge(left=country, right=df[['pop', 'GID_0', 'bs', 'year', 'index']], on='GID_0')
    y = x.groupby(['GID_0', 'year']).agg({'pop': 'sum', 'bs': 'sum', 'index': 'mean'}).reset_index()
    z = pd.merge(y, country)
    z['type'] = 'division'
    z['GID_0'] = 'WWW'
    gpd.GeoDataFrame(z).to_file(file, driver='GeoJSON', index=False)
