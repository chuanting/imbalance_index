# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 01_grid_level_index.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-18 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import CRS
import geojson
import mobile_codes
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray
import time
import os


def split_poly_into_grids(poly, x_size, y_size):
    poly = poly.explode().reset_index()
    poly['area'] = 0
    poly.loc[:, 'area'] = poly['geometry'].area
    poly = poly.sort_values(by='area', ascending=False)
    grid_index = 0
    polygons = []
    for k in range(poly.shape[0]):
        poly_inside = poly.iloc[k]
        lon_min, lat_min, lon_max, lat_max = poly_inside['geometry'].bounds
        whole_poly = Polygon([[lon_min, lat_max],
                              [lon_max, lat_max],
                              [lon_max, lat_min],
                              [lon_min, lat_min],
                              [lon_min, lat_max]])

        whole_poly = gpd.GeoSeries(whole_poly)
        # whole_poly.crs = CRS("epsg:4326")

        xmin, ymin, xmax, ymax = whole_poly.total_bounds

        # height = (ymax - ymin) / rows
        # width = (xmax - xmin) / cols
        rows = int((ymax - ymin) / y_size)
        cols = int((xmax - xmin) / x_size)

        if rows == 0:
            rows = 1
        if cols == 0:
            cols = 1
        x_left = xmin
        x_right = xmin + x_size
        y_top = ymax
        y_bot = ymax - y_size

        if k >= 2:
            break

        for i in range(cols):
            y_top_temp = y_top
            y_bot_temp = y_bot
            for j in range(rows):
                p = Polygon([(x_left, y_top_temp),
                             (x_right, y_top_temp),
                             (x_right, y_bot_temp),
                             (x_left, y_bot_temp)])
                polygons.append(geojson.Feature(geometry=p, properties={"id": grid_index}))
                y_top_temp = y_top_temp - y_size
                y_bot_temp = y_bot_temp - y_size

                grid_index += 1
            x_left = x_left + x_size
            x_right = x_right + x_size

    fc_grid = geojson.FeatureCollection(polygons)

    return fc_grid


def get_info_per_grid(bs, pop, poly, grid):
    start = time.time()
    print('Creating GeoDataFrame from grid data...')
    df_bounds = grid['geometry'].bounds
    x = (df_bounds['minx'] + df_bounds['maxx']) / 2
    y = (df_bounds['miny'] + df_bounds['maxy']) / 2
    grid['lon'] = x.values
    grid['lat'] = y.values

    # grid.set_crs('epsg:4326', inplace=True, allow_override=True)
    gpd_grid = GeoDataFrame(grid)

    df_country_geo = GeoDataFrame({'geometry': PointArray(gpd_grid[['lon', 'lat']].values.tolist()),
                                   'id': gpd_grid['id']})
    df_country_grid = sjoin(df_country_geo, poly)

    gpd_grid = gpd_grid.loc[df_country_grid.index]
    print('Time: {:}'.format(time.time() - start))

    print('Creating GeoDataFrame from population data...')
    start = time.time()
    gdf_population = GeoDataFrame({'pop': pop['population'],
                                   'geometry': PointArray(pop[['longitude', 'latitude']].values.tolist())})
    print('Time: {:}'.format(time.time() - start))

    start = time.time()
    print('Creating GeoDataFrame from bs data...')
    gdf_bs = GeoDataFrame({'geometry': PointArray(bs[['lon', 'lat']].values.tolist()),
                           'bs': 1.0, 'date': bs['created']})
    print('Time: {:}'.format(time.time() - start))

    print('Calculating population per grid')
    start = time.time()
    pop_per_grid = sjoin(gdf_population, gpd_grid)
    pop_per_grid_group = pop_per_grid.groupby('id').sum().reset_index()[['id', 'pop']]
    print('Time: {:}'.format(time.time() - start))

    print('Calculating # of BS per grid')
    start = time.time()
    results = []
    for year in range(2014, 2023):
        print(year)
        dates = pd.to_datetime(str(year))
        unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]

        bs_per_grid = sjoin(bs_year, gpd_grid)
        bs_per_grid_group = bs_per_grid.groupby('id').count().reset_index()[['id', 'index_right']]
        bs_per_grid_group.columns = ['id', 'bs']
        # print('Time: {:}'.format(time.time() - start))

        print('Align BS and grid')
        start = time.time()
        pop_align = pd.merge(left=pop_per_grid_group, right=gpd_grid, on='id', how='right')
        bs_align = pd.merge(left=bs_per_grid_group, right=gpd_grid, on='id', how='right')

        pop_align['pop'].fillna(0, inplace=True)
        bs_align['bs'].fillna(0, inplace=True)
        # print('Time: {:}'.format(time.time() - start))

        pop_align['bs'] = bs_align['bs']
        pop_align['year'] = year
        pop_align['index'] = imbalance_index(pop_align['pop'], pop_align['bs'])
        results.append(pop_align)

    print('Saving to local disk')
    all_info = pd.concat(results)
    return all_info


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
    folder = 'data/'
    save_folder = 'data/final/grid/'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    poly_file = folder + 'gadm36_shp/country-level-4-digit.geojson'
    print('Loading worldwide boundary file')
    worldwide = gpd.read_file(poly_file)
    # country_poly = worldwide.loc[worldwide['GID_0']=='WWW']
    df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    country_names = 'data/MCI_Data_2020.xls'
    df_country_info = pd.read_excel(country_names, skiprows=2, sheet_name=2)
    df_name = df_country_info.loc[df_country_info['Year'] == 2019]
    for i, name in enumerate(df_name['ISO Code']):
        print('Now processing {:} data'.format(name))

        # if name != 'BRB':
        #     continue
        if i > 2:
            break
        mcc = np.array([mobile_codes.alpha3(name).mcc], dtype=int).ravel()
        country_bs = df_bs.loc[df_bs['mcc'].isin(mcc)]
        country_pop_file = 'data/{:}.gz'.format(name)

        # load population data
        if not os.path.exists(country_pop_file):
            print('File not exists')
            continue

        df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
        df_poly = worldwide.loc[worldwide['GID_0'] == name]

        if df_poly.shape[0] == 0:
            continue

        x_size, y_size = 0.3, 0.2
        country_grid = split_poly_into_grids(df_poly, x_size, y_size)

        df_poly = GeoDataFrame(df_poly)
        df_grid = gpd.GeoDataFrame.from_features(country_grid['features'])

        # get the number of BS and population per grid
        start = time.time()
        all_info = get_info_per_grid(country_bs, df_pop, df_poly, df_grid)
        if all_info.shape[0] == 0:
            continue
        all_info['GID_0'] = name
        all_info = GeoDataFrame(all_info)
        all_info.to_geopandas().to_file(save_folder + name + '.grid_{:}_{:}.geojson'.format(x_size, y_size), driver='GeoJSON')


if __name__ == '__main__':
    main()
