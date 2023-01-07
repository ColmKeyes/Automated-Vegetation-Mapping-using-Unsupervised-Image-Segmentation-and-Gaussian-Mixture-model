import argparse
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from skimage import metrics
import geopandas as gpd
import rasterio as rasta
import numpy as np
from sklearn.decomposition import PCA
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import rasterio.mask
import fiona
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.plot import show
from rasterio.enums import MergeAlg
import spectral as spy

## TODO
##  - get labelled principle components of GT data, aka plot the principle components & label the pixels within the PCs with which GT cluster they are in,

#########################
## Some Bounding Box info
#########################
xsize= (16537 - 16392)*(100/8)
ysize= (36870 - 36745)*(100/8)
print(xsize)
print(ysize)

#coordinates of the boundingbox
x1=163920
y1=368700
x2=165370
y2=367450
#by using the corners of the boundingbox as Ground control points, the image can be georeferenced:
#setting GCPs
tl = GroundControlPoint(0, 0, x1, y1)
bl = GroundControlPoint(1666.6666666666667, 0, x1, y2)#1666, 0, x1, y2)
br = GroundControlPoint(1666.6666666666667, 1933.3333333333335, x2, y2)
tr = GroundControlPoint(0, 1933.3333333333335, x2, y1)
gcps = [tl, bl, br, tr]

## Affine trasnform is a simple total trasnsform wehich will transform each point on a 2d axis in each direction, which is given by a 3x3 array for left right lower left lower right etc.
## our transform will be completed using ground points to transform the points to an exact pixel position.
#transform to GTIFF
transform = from_gcps(gcps)
crs = rasterio.crs.CRS.from_epsg(28992)

groundtruth_shp = "C:/Users/Colm The Creator/PycharmProjects/UISB/data/GT/ACT_Final_Groundtruth.shp"
HRtif= rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_8cm.tif")

groundtruth = gpd.read_file(groundtruth_shp)

groundtruth['coords'] = groundtruth['geometry'].apply(lambda x: x.representative_point().coords[:])
groundtruth['coords'] = [coords[0] for coords in groundtruth['coords']]

# Get list of geometries for all features in vector file
geom = [shapes for shapes in groundtruth.geometry]


# create a numeric unique value for each GT class
groundtruth['u_label'] = groundtruth.groupby(['Label']).ngroup()

## Mask only Rendiermoss as an example
masked_polys = rasta.mask.mask(HRtif,groundtruth[groundtruth['Label'] == 'Rendiermos'].geometry)

orthoHR_lan = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_8cm.lan")
orthoHR_lan = orthoHR_lan.load()

## set all masked pixels to class 1
masked_polys[0][0][np.nonzero(masked_polys[0][0])] = 1



#############################################################################
## Spectral Angle Mapper
#############################################################################

## create training class loader for SpectralPy
Rendiermos_spy_training_class = spy.create_training_classes(orthoHR_lan,masked_polys[0][0],True)

## we must first create a CxB array of training class mean spectra,
## where C is the number of training classes and B is the number of spectral bands.
means = np.zeros((len(Rendiermos_spy_training_class), orthoHR_lan.shape[2]), float)

for (i, c) in enumerate(Rendiermos_spy_training_class):
    means[i] = c.stats.mean

## Calculate Spectral Angle
angles = spy.spectral_angles(orthoHR_lan, means)

## Here, the minimum angles are the ones closest to the endmember's spectra,
## But this seems to be off a small bit(0.2), possibly because our GT data is not exact?

plt.imshow(angles.clip(0.15,0.25))
spy.imshow(classes=angles.clip(0.15,0.25))
#plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
rasta.plot.show(HRtif, ax=ax)
groundtruth[groundtruth['Label'] == 'Rendiermos'].plot(ax=ax, facecolor='none', edgecolor='red')
plt.show()














#############################################################################
## Gaussian Maximum Likelihood CLassification
#############################################################################
# gmlc = spy.GaussianClassifier(Rendiermos_spy_training_class)
#
# clmap = gmlc.classify_image(orthoHR_lan)
#
# spy.imshow(classes=clmap)





#############################################################################
## PCA (discontinued for now..)
#############################################################################


# pca = PCA(3)
#
# ## split image into rgb
# r,g,b = Rendiermos_image[:]
#
# ## first 10 Principle components of the whole GT image. These will contain the
#
# r_pca = pca.fit_transform(r)
# g_pca = pca.fit_transform(g)
# b_pca = pca.fit_transform(b)

#plt.scatter(GT_pca[:,0],GT_pca[:,1])
#plt.show()

spy.im



##################################################################

# ## some plotting
#
##################################################################

# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize = (10,6))
#
# c_map = plt.cm.get_cmap('jet', 10)
# #plt.scatter(GT_pca[:, 0], GT_pca[:, 1], s = 15,
# #            cmap = c_map )#, c = digits.target)
# #plt.colorbar()
# #plt.xlabel('PC-1') , plt.ylabel('PC-2')
# #plt.show()

# plt.scatter(r_pca[:, 0], r_pca[:, 1],c='r')
# plt.scatter(g_pca[:, 0], g_pca[:, 1],c='g')
# plt.scatter(b_pca[:, 0], b_pca[:, 1],c='b')
# plt.show()