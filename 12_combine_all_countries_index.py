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


folder = 'data/final/'
results = []
for i, file in enumerate(glob.glob(folder + '*.geojson')):
    print(file)
    # if i > 2:
    #     break
    country = gpd.read_file(file)
    results.append(country)

other_sources = gpd.read_file('data/gadm36_shp/world_all_levels_pop_v2.geojson')
www = other_sources.loc[other_sources['GID_0']=='WWW']
results.append(www)
final_index = pd.concat(results)
final_index.to_file('d:/final_index.geojson', driver='GeoJSON', index=False)
# gpd.GeoDataFrame(final_index).to_file()