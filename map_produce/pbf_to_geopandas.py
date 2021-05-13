# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: pbf_to_geopandas.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-05-02 (YYYY-MM-DD)
-----------------------------------------------
"""
from pyrosm import get_data
from pyrosm.data import sources
from pyrosm import OSM
import matplotlib.pyplot as plt
import geopandas as gpd

save_folder = '../data/osm/'
# The first call won't download the data because it was already downloaded earlier
fp = get_data("Afghanistan", directory=save_folder)
print("Data was downloaded to:", save_folder)

print(fp)

osm = OSM(fp)

# Read all boundaries using the default settings
boundaries = osm.get_boundaries()
gdf_bound = gpd.GeoDataFrame(boundaries)
print(gdf_bound.head())
gdf_bound.to_csv('d:/test.csv', index=False)
# boundaries.plot(facecolor="none", edgecolor="blue")

# fp = get_data("test_pbf")
# # Initialize the OSM parser object
# osm = OSM(fp)
# buildings = osm.get_buildings()
# buildings.plot()
# plt.show()

# afghanistan
url = "https://osm-boundaries.com/Download/Submit?apiKey=5b4605267dafc4f9c94d5622f884457f&db=osm20210315&osmIds=-303427&minAdminLevel=2&maxAdminLevel=13"

# albania
url = "https://osm-boundaries.com/Download/Submit?apiKey=5b4605267dafc4f9c94d5622f884457f&db=osm20210315&osmIds=-53292&minAdminLevel=2&maxAdminLevel=13"