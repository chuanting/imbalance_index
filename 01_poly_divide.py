# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: 01_poly_divide.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2021-03-14 (YYYY-MM-DD)
-----------------------------------------------
"""
import geopandas as gpd
from shapely.geometry import Polygon
import geojson
from pyproj import CRS

rows, cols = 100, 300

folder = 'D:/Dataset/HighResolutionPopulation/'
country = 'Tunisia'

# obtain the following info through https://mygeodata.cloud/conversion
poly_file = folder+country+'/poly/poly.shp'
# country_border = gpd.read_file('D:/Dataset/HighResolutionPopulation/Tunisia/Tunisia_Bound.shp')
country_border = gpd.read_file(poly_file)
country_border.to_file(folder+country+'/poly.geojson', driver='GeoJSON')
lon_min, lat_min, lon_max, lat_max = country_border['geometry'][0].bounds
#

city_poly = Polygon([[lon_min, lat_max],
                     [lon_max, lat_max],
                     [lon_max, lat_min],
                     [lon_min, lat_min],
                     [lon_min, lat_max]])

geo_poly = gpd.GeoSeries(city_poly)

geo_poly.crs = CRS("epsg:4326")

# geo_poly.to_crs(epsg=3857)

xmin, ymin, xmax, ymax = geo_poly.total_bounds

height = (ymax - ymin) / rows
width = (xmax - xmin) / cols
x_left = xmin
x_right = xmin + width
y_top = ymax
y_bot = ymax - height
polygons = []
for i in range(cols):
    y_top_temp = y_top
    y_bot_temp = y_bot
    for j in range(rows):
        p = Polygon([(x_left, y_top_temp),
                     (x_right, y_top_temp),
                     (x_right, y_bot_temp),
                     (x_left, y_bot_temp)])
        polygons.append(geojson.Feature(geometry=p, properties={"id": i*cols + j}))
        y_top_temp = y_top_temp - height
        y_bot_temp = y_bot_temp - height
    x_left = x_left + width
    x_right = x_right + width

fc_grid = geojson.FeatureCollection(polygons)
fc_poly = geojson.FeatureCollection(geojson.Feature(geometry=city_poly))
with open(folder+country+'/grid.geojson', 'w') as f:
    geojson.dump(fc_grid, f)