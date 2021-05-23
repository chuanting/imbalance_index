# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 00_calc_bs.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-23 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
import spatialpandas as spd
import pandas as pd
import time
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray

#
gpd_pop = gpd.read_file('../../data/gadm36_shp/city-level-pop.geojson')
print(gpd_pop.head())

print('Loading BS data')
df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
# df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_1_percent.csv')
df_bs = df_bs.sample(frac=0.1)

start = time.time()
print('Creating GeoDataFrame from bs data...')
gdf_bs = GeoDataFrame({'geometry': PointArray(df_bs[['lon', 'lat']].values.tolist()), 'bs': 1.0, 'date': df_bs['created']})
print('Time: {:}'.format(time.time() - start))
gdf_bs.to_geopandas().to_file('../../data/gadm36_shp/bs-01.shp')

# for year in range(2011, 2023):
#     dates = pd.to_datetime(str(year))
#     unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#     bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]
#
#     pop_bs = sjoin(bs_year, GeoDataFrame(gpd_pop))
#     print(pop_bs.head())
