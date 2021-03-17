# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: combine_geojson.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-16 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
import pandas as pd
import glob

files = glob.glob('data/*index.geojson')
index_list = []
for i, file in enumerate(files):
    if i > 20:
        break
    print(file)
    index_list.append(gpd.read_file(file))

print('concat')
final_index = gpd.GeoDataFrame(pd.concat(index_list))
final_index.to_file('d:/final.index.geojson', driver='GeoJSON')

