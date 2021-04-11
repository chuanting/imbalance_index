# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 02_calc_bs_pop.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-14 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
import json
import geopandas as gpd
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import Polygon, PointArray, MultiPolygon


folder = 'D:/Dataset/HighResolutionPopulation/'
country = 'Tunisia'
pop_file = folder + country + '/pop.csv'
bs_file = folder + country + '/BS.csv'
poly_file = folder + country + '/poly/poly.shp'
grid_file = folder + country + '/grid.geojson'

print('Loading population, bs, poly, and grid data...')
df_pop = pd.read_csv(pop_file)
df_bs = pd.read_csv(bs_file)
gdf_poly = gpd.read_file(poly_file)
gdf_poly = GeoDataFrame(gdf_poly)
# gdf_poly.set_crs('epsg:4326', inplace=True, allow_override=True)
grid = json.load(open(grid_file))
print('Creating GeoDataFrame from grid data...')

df_grid = gpd.GeoDataFrame.from_features(grid['features'])
df_grid['id'] = range(df_grid.shape[0])
df_grid.set_crs('epsg:4326', inplace=True, allow_override=True)
df_grid = GeoDataFrame(df_grid)
print(df_grid)

print('Creating GeoDataFrame from population data...')
# gdf_population = gpd.GeoDataFrame(df_pop,
#                                   geometry=gpd.points_from_xy(df_pop.Lon, df_pop.Lat))
gdf_population = GeoDataFrame({'Population': df_pop['Population'],
                               'geometry': PointArray(df_pop[['Lon', 'Lat']].values.tolist())})
# gdf_population.set_crs('epsg:4326', inplace=True, allow_override=True)
print('Creating GeoDataFrame from bs data...')
# gdf_bs = gpd.GeoDataFrame(df_bs,
#                           geometry=gpd.points_from_xy(df_bs.lon, df_bs.lat))
gdf_bs = GeoDataFrame({'geometry': PointArray(df_bs[['lon', 'lat']].values.tolist()),
                       'radio': df_bs['radio'], 'net': df_bs['net'],
                       'cell': df_bs['cell'], 'lon': df_bs['lon'],
                       'lat': df_bs['lat']})
# gdf_bs.set_crs('epsg:4326', inplace=True)

print('Calculating # of populations per grid...')
pop_in_grid = sjoin(gdf_population, df_grid)
# pop_in_grid = gpd.sjoin(gdf_population, df_grid, op='within')
print('Calculating # of bs per grid...')
# bs_in_grid = gpd.sjoin(gdf_bs, df_grid, op='within')
bs_in_grid = sjoin(gdf_bs, df_grid)
# print(bs_in_grid.head())

bs_in_grid.drop(['index_right'], axis=1, inplace=True)

print('Removing the BS that are out of the poly...')
# bs_in_grid = gpd.sjoin(bs_in_grid, gdf_poly, op='within')
bs_in_grid = sjoin(bs_in_grid, gdf_poly)


print(bs_in_grid.head(10))
bs_in_grid = bs_in_grid[['radio', 'net', 'cell', 'lon', 'lat', 'geometry', 'id_left']]

bs_all_count = bs_in_grid[['id_left', 'geometry']].groupby('id_left').count().reset_index()
print(bs_all_count.head(10))
bs_all_count.columns = ['id', 'bs']
# bs_all_count = GeoDataFrame(bs_all_count)

pop_in_grid = pop_in_grid[['id', 'geometry']]
pop_count = pop_in_grid.groupby('id').count().reset_index()
pop_count.columns = ['id', 'pop']
# pop_count = GeoDataFrame(pop_count)
print(pop_count.head(10))
print(df_grid.head(10))

print('Population and BS alignment...')
# pop_align = sjoin(pop_count, df_grid)
# bs_align = sjoin(bs_all_count, df_grid)
pop_align = pd.merge(left=pop_count, right=df_grid, left_on='id', right_on='id', how='right')
bs_align = pd.merge(left=bs_all_count, right=df_grid, left_on='id', right_on='id', how='right')

print(bs_align.head())

# pop_align.fillna(0, inplace=True)
# bs_align.fillna(0, inplace=True)
pop_align.columns = ['id', 'pop', 'geometry']
bs_align.columns = ['id', 'bs', 'geometry']

pop_align['pop'] = pop_align['pop'].fillna(0)
bs_align['bs'] = bs_align['bs'].fillna(0)
print(pop_align.head())
print(bs_align.head())
# pop_results = gpd.GeoDataFrame(pop_align)
pop_results = GeoDataFrame(pop_align)

# pop_results.set_crs('epsg:4326', inplace=True)
# pop_final = gpd.sjoin(pop_results, gdf_poly, op='within')[['id', 'pop', 'geometry']]
pop_final = sjoin(pop_results, gdf_poly)
print(pop_final.head(10))

# [['id', 'pop', 'geometry']]
# pop_final = pop_align
# pop_final = GeoDataFrame(pop_align)

df_bounds = pop_final['geometry'].bounds
x = (df_bounds['x0'] + df_bounds['x1']) / 2
y = (df_bounds['y0'] + df_bounds['y1']) / 2
pop_final['lon'] = x.values
pop_final['lat'] = y.values
pop_final.to_geopandas().to_file(folder + country + '/pop.geojson', driver='GeoJSON')

# bs_results = gpd.GeoDataFrame(bs_align)
# bs_results = GeoDataFrame(bs_align)
# bs_results.set_crs('epsg:4326', inplace=True)
# bs_final = gpd.sjoin(bs_results, gdf_poly, op='within')[['id', 'bs', 'geometry']]
# bs_final = sjoin(bs_results, gdf_poly)[['id', 'bs', 'geometry']]

bs_final = GeoDataFrame(bs_align)
bs_final['lat'] = x.values
bs_final['lon'] = y.values
bs_final.to_geopandas().to_file(folder + country + '/bs.geojson', driver='GeoJSON')

pop_final['bs'] = bs_final['bs'].values
all_final = pop_final.copy()
all_final.to_geopandas().to_file(folder + country + '/all.geojson', driver='GeoJSON')
