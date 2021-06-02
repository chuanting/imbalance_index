## On Telecommunication Service Imbalance

Source code to generate the imbalance index of each country


## prepare python env
```
sh creat_env
```

## update python env
```
conda env update --prefix ./env --file environment.yml  --prune
```


## Generate US poly

1. go to http://www.diva-gis.org/gdata and download the country shapefile
2. drag the admin1 or admin2 to QGIS
3. select feature and paste as new vector layer
4. edit-->merge selected features-->save

## Data Prepare

We use three kinds of data, i.e., population data, BS data, and worldwide polygons. 
The population data, which is originally released by 
FaceBook and can be accessed through 
[Humdata](https://data.humdata.org/organization/facebook?q=High%20Resolution%20Population) or through
[AWS](https://registry.opendata.aws/dataforgood-fb-hrsl/).
The BS data is released by [OpenCellId Project](https://www.opencellid.org/).
The worldwide polygons are downloaded from [GADM](https://gadm.org/data.html).

There are two version of imbalance index, i.e., grid-level and (sub-)division-level.
## Grid-level imbalance index

Given a country name in the format of  [ISO 3166-1 alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3), we first
find the country polygon and calculate the total bounds of this polygon. Then we split the country into M * N 
grids/granules. After that, we count the number of BSs and total populations inside each grid. Finally we calculate
the imbalance index using our proposed formula.

## Division-level imbalance index
The process to obtain the division-level index is quite similar to the grid-level one, but with one difference. That is,
we do not split the country polygon manually, instead we use the official division boundary. We calculate the number of
BSs and total populations inside each division. We calculate the imbalance index using our proposed formula.

## important notice

https://cengel.github.io/R-spatial/mapping.html#choropleth-mapping-with-spplot

https://www.r-graph-gallery.com/327-chloropleth-map-from-geojson-with-ggplot2.html

https://plotly.com/r/choropleth-maps/

https://rstudio-pubs-static.s3.amazonaws.com/324400_69a673183ba449e9af4011b1eeb456b9.html



## reduce GeoJSON complexity

https://github.com/mbloch/mapshaper 

1. Install npm first.
2. Use the following commands to reduce the size of the file.

```shell script
mapshaper -i gadm36.shp -simplify 0.1% keep-shapes -filter-islands min-area=100km2 remove-empty -filter 'GID_0=="ATA"' invert -o force simplified-world.shp
```

3. Merge polygons into a country-level geojson

```shell script
mapshaper simplified-world.shp -simplify 1% -dissolve GID_0 -o format=geojson country-level.geojson
```
```shell script
mapshaper simplified-world.shp -simplify 1% -dissolve NAME_1,NAME_2,NAME_3 -o format=geojson three-levels-01.geojson
```
4. Remove places with very small area and also the Antarctica 

```python
import geopandas as gpd
world = gpd.read_file('country-level.geojson')
world['area'] = world.geometry.area
world.drop(world.loc[world['GID_0']=='ATA'].index, inplace=True)
filtered = world.loc[world['area']>=2]
filtered.to_file('filtered_world.geojson', driver='GeoJSON')
```

5. Count number of points in a polygon
```shell script
mapshaper three-levels-01.geojson -join pop.geojson calc='pop=count()' fields= -o force format=geojson a.geojson

mapshaper-xl 50gb all_name.geojson -join pop.geojson calc='pop=count()' fields= -o force format=geojson city-level-pop.geojson
```

## convert from geojson to mbtiles
1. on you mac, type xcode-select --install to upate your homebrew
2. brew install tippecanoe
3. ```shell script
tippecanoe -zg -o out.mbtiles --drop-densest-as-needed in.geojson

```shell script
ogr2ogr -f GEOJSON results.geojson world.pop.csv -oo X_POSSIBLE_NAMES=longitude -oo Y_POSSIBLE_NAMES=latitude -oo KEEP_GEOM_COLUMNS=NO
```


## Change name
```shell script
for file in *.geojson; do mv "$file" "${file/grid_0.4_0.4.}"; done
```

## tippecanoe batch
```shell script
for file in *.geojson; do tippecanoe -z8 -f --layer=grid -o "$file".mbtiles "$file"; done
```