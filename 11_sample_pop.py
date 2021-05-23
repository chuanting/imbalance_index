# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 11_sample_pop.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-22 (YYYY-MM-DD)
-----------------------------------------------
"""
import time
import glob
import pandas as pd
import geopandas as gpd
from spatialpandas import GeoDataFrame, sjoin
from spatialpandas.geometry import PointArray


def main():
    all_country = []
    frac = 0.01
    scale = 1. / frac
    print('Loading worldwide population data...')
    for i, file in enumerate(glob.glob('data/*[!pop].gz')):
        print(file)
        # if i >= 5:
        #     break
        df_country = pd.read_csv(file, compession='gzip', sep='\t', header=0)
        df_sample = df_country.sample(frac=frac)
        all_country.append(df_sample)

    df_pop = pd.concat(all_country, ignore_index=True)

    print('Creating GeoDataFrame from population data...')
    start = time.time()
    gdf_population = GeoDataFrame({'pop': df_pop['population']*scale,
                                   'geometry': PointArray(df_pop[['longitude', 'latitude']].values.tolist())})
    # # gdf_population.crs
    # print(gdf_population.shape)
    print('Time: {:}'.format(time.time() - start))
    #
    gdf_pop = gdf_population.to_geopandas()
    gdf_pop.crs
    print('Saving to disk...')
    start = time.time()
    # # gdf_pop.to_file('d:/test.shp')
    gdf_pop.to_file('d:/test.geojson', driver='GeoJSON')
    print('Time: {:}'.format(time.time() - start))

    # print('Loading worldwide boundary file')
    # gdf_geo = gpd.read_file('data/gadm36_shp/three-levels-001.geojson')
    # # BS file
    # print('Loading BS data')
    # df_bs = pd.read_csv('D:/Dataset/cell_towers/cell_towers_2020-12-08-T000000.csv')
    # df_bs = df_bs.sample(frac=frac)
    #
    # start = time.time()
    # print('Creating GeoDataFrame from bs data...')
    # gdf_bs = GeoDataFrame({'geometry': PointArray(df_bs[['lon', 'lat']].values.tolist()),
    #                        'bs': 1.0 * scale})
    # print('Time: {:}'.format(time.time() - start))
    #
    # print('Calculating population per grid')
    # start = time.time()
    # pop_per_grid = sjoin(gdf_population, gdf_geo)
    # pop_per_grid_group = pop_per_grid.groupby('UID').sum().reset_index()[['UID', 'pop']]
    # print('Time: {:}'.format(time.time() - start))
    # #
    # #
    # # # gdf[['population', 'geometry']].to_file('../data/pop_sample_01.geojson', driver='GeoJSON', index=False)
    # # gdf.to_file('pop_sample_01.geojson', driver='GeoJSON', index=False)
    # #
    # # final_df.to_csv('data/pop_sample_01.csv', index=False)


if __name__ == '__main__':
    main()