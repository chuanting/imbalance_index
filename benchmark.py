# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: benchmark.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-15 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas
from spatialpandas import GeoSeries, GeoDataFrame
from spatialpandas import sjoin
from geopandas import sjoin as gp_sjoin
import pandas as pd
import dask.dataframe as dd
import timeit

from dask.distributed import Client, LocalCluster


if __name__ == '__main__':
    cluster = LocalCluster(dashboard_address=':8791',
                           n_workers=16,
                           threads_per_worker=1,
                           memory_limit='5 GB')
    client = Client(cluster)

    world_gp = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    world_gp = world_gp.sort_values('name').reset_index(drop=True)
    world_gp = world_gp[['pop_est', 'continent', 'name', 'geometry']]
    print(world_gp)
    world_df = GeoDataFrame(world_gp)

    cities_gp = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_cities')
    )
    cities_df = GeoDataFrame(cities_gp)
    print(cities_df.head())

    reps = 500000
    # Large geopandas GeoDataFrame
    cities_large_gp = pd.concat([cities_gp] * reps, axis=0)

    # Large spatialpandas GeoDataFrame
    cities_large_df = pd.concat([cities_df] * reps, axis=0)

    # Large spatialpandas DaskGeoDataFrame with 16 partitions
    cities_large_ddf = dd.from_pandas(cities_large_df, npartitions=16).persist()

    # Precompute the partition-level spatial index
    cities_large_ddf.partition_sindex

    print("Number of Point rows: %s" % len(cities_large_df))
    print("Number of MultiPolygon rows: %s" % len(world_df))

    # %%timeit
    import time

    # start = time.time()
    # print(len(gp_sjoin(cities_large_gp, world_gp)))
    # print('Time: {:}'.format(time.time() - start))
    start = time.time()
    print(len(sjoin(cities_large_df, world_df)))
    print('Time: {:}'.format(time.time() - start))
    start = time.time()
    print(len(sjoin(cities_large_ddf, world_df)))
    print('Time: {:}'.format(time.time() - start))