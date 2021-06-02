#!/bin/bash

for file in grid/*.geojson;
do tippecanoe -z8 -f -o "$file".mbtiles "$file"
done
