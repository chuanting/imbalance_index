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