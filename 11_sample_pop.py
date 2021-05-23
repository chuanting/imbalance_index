# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 11_sample_pop.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-22 (YYYY-MM-DD)
-----------------------------------------------
"""
import glob
import pandas as pd
import geopandas as gpd


def main():
    all_country = []
    for file in glob.glob('data/*[!pop].gz'):
        print(file)
        df_country = pd.read_csv(file, compression='gzip', sep='\t', header=0)
        df_sample = df_country.sample(frac=0.1)
        all_country.append(df_sample)

    final_df = pd.concat(all_country, ignore_index=True)

    gdf = gpd.GeoDataFrame(final_df, geometry=gpd.points_from_xy(final_df.longitude, final_df.latitude))
    # gdf[['population', 'geometry']].to_file('../data/pop_sample_01.geojson', driver='GeoJSON', index=False)
    gdf.to_file('pop_sample_01.geojson', driver='GeoJSON', index=False)

    final_df.to_csv('data/pop_sample_01.csv', index=False)


if __name__ == '__main__':
    main()