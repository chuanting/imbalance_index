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


# division_folder = 'data/final/division/'
# division_results = []
# for i, file in enumerate(glob.glob(division_folder + '*.geojson')):
#     print(file)
#     country = gpd.read_file(file)
#     division_results.append(country)
#
# df_division = pd.concat(division_results, ignore_index=True)
# df_division['type'] = 'division'
# df_division.to_file('data/final/division.geojson', driver='GeoJSON')
#
df_division = gpd.read_file('data/final/division.geojson')

grid_folder = 'data/final/grid/'
grid_results = []
for i, file in enumerate(glob.glob(grid_folder + '*.geojson')):
    print(file)
    # if i > 5:
    #     break
    country = gpd.read_file(file)
    grid_results.append(country)
df_grid = pd.concat(grid_results, ignore_index=True)
df_grid.drop(['lon', 'lat'], inplace=True, axis=1)
df_grid['type'] = 'grid'
df_grid.to_file('data/final/grid.geojson', driver='GeoJSON')

final_index = pd.concat([df_division, df_grid], ignore_index=True)
final_index.to_file('d:/final_index.geojson', driver='GeoJSON', index=False)