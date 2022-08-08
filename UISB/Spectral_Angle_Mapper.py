"""
@Time    : 01/08/2022 12:34
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : Spectral_Angle_Mapper.py
@License : MIT
"""

#############################################################################
## Spectral Angle Mapper -- DL Validation Approach
#############################################################################

import geopandas as gpd
import rasterio as rasta
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
import spectral as spy

## TODO
##  - Build in an automated validation % to UISB, aka add the SAM into the DL loss function.
##  - add a comparison between spectral class and DL class output,
##      -- map the DL class-spectral class comparison to a contingency table,
##      -- possibly add numeric probabilities for each class, what threshold will be needed for a class to be correctly labelled?
##  - Expand this script for all classes,

groundtruth_shp = "C:/Users/Colm The Creator/PycharmProjects/UISB/data/GT/ACT_Final_Groundtruth.shp"
groundtruth = gpd.read_file(groundtruth_shp)
HRtif= rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.tif")

groundtruth['coords'] = groundtruth['geometry'].apply(lambda x: x.representative_point().coords[:])
groundtruth['coords'] = [coords[0] for coords in groundtruth['coords']]

# Get list of geometries for all features in vector file
geom = [shapes for shapes in groundtruth.geometry]

# create a numeric unique value for each GT class
groundtruth['u_label'] = groundtruth.groupby(['Label']).ngroup()

## Mask only Rendiermoss as an example
masked_polys = rasta.mask.mask(HRtif,groundtruth[groundtruth['Label'] == 'Rendiermos'].geometry)

#############################################################################
## Conversion from .tif to .lan
##  .LAN sensor data type, for reading into SpectralPy, used for spectral analysis in python.
## this is preferably done through terminal vs in python due to GDAL's stability in terminal
#############################################################################
## use this for file conversion in terminal:                 gdal_translate -of LAN orthoHR_75mm.tif orthoHR_75mm.lan
#############################################################################
orthoHR_lan = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.lan")
orthoHR_lan = orthoHR_lan.load()

## set all masked pixels to class 1
masked_polys[0][0][np.nonzero(masked_polys[0][0])] = 1

#############################################################################
## Spectral Angle Mapper        Rendiermos_example
#############################################################################

## create training class loader for SpectralPy
Rendiermos_spy_training_class = spy.create_training_classes(orthoHR_lan,masked_polys[0][0],True)

## we must first create a CxB array of training class mean spectra, C = training class, B = spectral band
means = np.zeros((len(Rendiermos_spy_training_class), orthoHR_lan.shape[2]), float)
for (i, c) in enumerate(Rendiermos_spy_training_class):
    ## mean RGB for Rendiermos
    means[i] = c.stats.mean

## Calculate Spectral Angle
angles = spy.spectral_angles(orthoHR_lan, means)

## Here, the minimum angles are the ones closest to the endmember's spectra,
## But this seems to be off a small bit(0.2), possibly because our GT data is not exact?

plt.imshow(angles.clip(0.15,0.25))
plt.title('Spectra Angle Map - Rendiermos')
fig, ax = plt.subplots(figsize=(15, 15))
rasta.plot.show(HRtif, ax=ax)
plt.title('orthoHR - Rendiermos')
groundtruth[groundtruth['Label'] == 'Rendiermos'].plot(ax=ax, facecolor='none', edgecolor='red')
view = spy.imshow(classes=angles.clip(0.15,0.25))
plt.title('Spectral Classes - Rendiermos')
## spy.imshow acts slightly weird compared to plt,
plt.pause(100)