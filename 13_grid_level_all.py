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
        rows = int(np.floor((ymax - ymin) / y_size))
        cols = int(np.floor((xmax - xmin) / x_size))

        while rows >= 250:
            y_size += 0.1
            rows = int(np.floor((ymax - ymin) / y_size))
        while cols >= 250:
            x_size += 0.1
            cols = int(np.floor((xmax - xmin) / x_size))

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

    start = time.time()
    print('Creating GeoDataFrame from bs data...')
    gdf_bs = GeoDataFrame({'geometry': PointArray(bs[['lon', 'lat']].values.tolist()),
                           'bs': 1.0, 'date': bs['created']})
    print('Time: {:}'.format(time.time() - start))

    print('Calculating population per grid')
    start = time.time()
    pop_per_grid = sjoin(pop, gpd_grid)
    pop_per_grid_group = pop_per_grid.groupby('id').sum().reset_index()[['id', 'pop']]
    print('Time: {:}'.format(time.time() - start))

    print('Calculating # of BS per grid')
    results = []
    for year in range(2011, 2023):
        print(year)
        dates = pd.to_datetime(str(year))
        unix_stamp = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        bs_year = gdf_bs.loc[gdf_bs['date'] <= unix_stamp]

        bs_per_grid = sjoin(bs_year, gpd_grid)
        bs_per_grid_group = bs_per_grid.groupby('id').count().reset_index()[['id', 'index_right']]
        bs_per_grid_group.columns = ['id', 'bs']
        # print('Time: {:}'.format(time.time() - start))
        pop_align = pd.merge(left=pop_per_grid_group, right=gpd_grid, on='id', how='right')
        bs_align = pd.merge(left=bs_per_grid_group, right=gpd_grid, on='id', how='right')

        pop_align['pop'].fillna(0, inplace=True)
        bs_align['bs'].fillna(0, inplace=True)
        # print('Time: {:}'.format(time.time() - start))

        pop_align['bs'] = bs_align['bs']
        pop_align['year'] = year
        pop_align['index'] = imbalance_index(pop_align['pop'], pop_align['bs'])

        pop_align['pop'] = pop_align['pop'].round(4)
        pop_align['index'] = pop_align['index'].round(4)
        pop_align.drop(['lon', 'lat'], axis=1, inplace=True)
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
    df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    country_names = worldwide['GID_0'].unique()
    for i, name in enumerate(country_names):
        print('Now processing {:} data'.format(name))
        try:
            codes = mobile_codes.alpha3(name)
            if codes.mcc is None:
                continue
        except Exception:
            print('Name {:} is not found'.format(name))
            continue

        mcc = np.array([codes.mcc], dtype=int).ravel()
        country_bs = df_bs.loc[df_bs['mcc'].isin(mcc)]
        country_pop_file = 'data/{:}.gz'.format(name)

        df_poly = worldwide.loc[worldwide['GID_0'] == name]
        if df_poly.shape[0] == 0:
            continue

        # load population data
        if not os.path.exists(country_pop_file):
            gdf_population = GeoDataFrame({'pop': 0, 'geometry': PointArray(df_poly.geometry.centroid)})
        else:
            df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
            gdf_population = GeoDataFrame({'pop': df_pop['population'],
                                           'geometry': PointArray(
                                               df_pop[['longitude', 'latitude']].values.tolist())})

        # df_pop = pd.read_csv(country_pop_file, compression='gzip', header=0, sep='\t')
        x_size, y_size = 0.4, 0.4
        country_grid = split_poly_into_grids(df_poly, x_size, y_size)

        df_poly = GeoDataFrame(df_poly)
        df_grid = gpd.GeoDataFrame.from_features(country_grid['features'])

        # get the number of BS and population per grid
        all_info = get_info_per_grid(country_bs, gdf_population, df_poly, df_grid)
        if all_info.shape[0] == 0:
            continue
        all_info['GID_0'] = name
        all_info = GeoDataFrame(all_info)
        all_info.to_geopandas().to_file(save_folder + name + '.grid_{:}_{:}.geojson'.format(x_size, y_size),
                                        driver='GeoJSON')


if __name__ == '__main__':
    main()
