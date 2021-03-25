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

