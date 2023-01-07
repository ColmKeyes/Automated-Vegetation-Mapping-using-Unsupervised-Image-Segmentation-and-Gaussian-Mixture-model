"""
@Time    : 01/08/2022 12:34
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Spectral_Angle_Mapper.py
@License : MIT
"""

#############################################################################
## Spectral Angle Mapper -- Validation Approach
#############################################################################

## NB, GDAL CAN BE FOUND WITHIN OSGEO!!!
from osgeo import gdal

import geopandas as gpd
import rasterio as rasta
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
import spectral as spy
import pandas as pd
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

#############
## Data inits
#############
stef_groundtruth_shp = 'C:/Users/Colm The Creator/Downloads/Stef_Polygons/Stef_Polygons.shp'
stef_groundtruth = gpd.read_file(stef_groundtruth_shp)
groundtruth_shp = "C:/Users/Colm The Creator/PycharmProjects/UISB/data/GT/ACT_Final_Groundtruth.shp"
groundtruth = gpd.read_file(groundtruth_shp)


HRtif = rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.tif")
HRtif = HRtif.read()
IRtif = rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/ortho25IR.tif")
## This tif is given in the form of CIR, coloured Infrared. with bands 1= IR, 2= green, 3= red
IRtif = IRtif.read(1)    ## READ TAKES NO BRACKETS FOR SOME REASON.. check api

combined_groundtruth_shp = 'C:/Users/Colm The Creator/PycharmProjects/UISB/combined_groundtruth.shp'
combined_groundtruth = gpd.read_file(combined_groundtruth_shp)
orthoHR_lan = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.lan")
orthoHR_lan = orthoHR_lan.load()
orthoIR_lan = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/ortho25IR.lan")
orthoIR_lan = orthoIR_lan.load()

HR_IR = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/HR_IR_stacked.lan").load()
## saving just first 4 layers..
HR_IR = HR_IR[:,:,:4]
#############





groundtruth['coords'] = groundtruth['geometry'].apply(lambda x: x.representative_point().coords[:])
groundtruth['coords'] = [coords[0] for coords in groundtruth['coords']]

# Get list of geometries for all features in vector file
geom = [shapes for shapes in groundtruth.geometry]

# create a numeric unique value for each GT class
groundtruth['u_label'] = groundtruth.groupby(['Label']).ngroup()

## Mask only Rendiermoss as an example
masked_polys = rasta.mask.mask(HRtif,groundtruth[groundtruth['Label'] == 'Rendiermos'].geometry)


groundtruth.plot('u_label') ## This takes into account the geometry value...


#############################################################################
## Conversion from .tif to .lan
##  .LAN sensor data type, for reading into SpectralPy, used for spectral analysis in python.
## this is preferably done through terminal vs in python due to GDAL's stability in terminal
#############################################################################
## use this for file conversion in terminal:                 gdal_translate -of LAN orthoHR_75mm.tif orthoHR_75mm.lan
#############################################################################