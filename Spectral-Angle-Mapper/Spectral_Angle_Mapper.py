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

#####
## Ignore annoying gpd warinings:
import warnings
warnings.filterwarnings('ignore')
#####
import geopandas as gpd
import rasterio as rasta
import rasterio.mask
import matplotlib.pyplot as plt
import numpy as np
from rasterio.plot import show
import spectral as spy
import pandas as pd
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import rasterio.plot

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
## TODO
##  - Build in an automated validation % to UISB, aka add the SAM into the DL loss function.
##  - add a comparison between spectral class and DL class output,
##      -- map the DL class-spectral class comparison to a contingency table,
##      -- possibly add numeric probabilities for each class, what threshold will be needed for a class to be correctly labelled?
##  - Expand this script for all classes,

# coordinates of the boundingbox
x1 = 163920
y1 = 368700
x2 = 165370
y2 = 367450
tl = GroundControlPoint(0, 0, x1, y1)
bl = GroundControlPoint(1666.6666666666667, 0, x1, y2)  # 1666, 0, x1, y2)
br = GroundControlPoint(1666.6666666666667,  1933.3333333333335, x2, y2)
tr = GroundControlPoint(0,  1933.3333333333335, x2, y1)
gcps = [tl, bl, br, tr]

transform = from_gcps(gcps)

xsize = (16537 - 16392) * (100 / 7.5)
ysize = (36870 - 36745) * (100 / 7.5)
print(xsize)
print(ysize)


#########################################
stef_groundtruth_shp = 'C:/Users/Colm The Creator/Downloads/Stef_Polygons/Stef_Polygons.shp'
stef_groundtruth = gpd.read_file(stef_groundtruth_shp)
groundtruth_shp = "C:/Users/Colm The Creator/PycharmProjects/UISB/data/GT/ACT_Final_Groundtruth.shp"
groundtruth = gpd.read_file(groundtruth_shp)
## Upon reading in the tif with rasta.open, we loose the coordinate information on it!!
## so to keep the crs data, we don't utilise rasta.read(), we use rasta.plot.show() on the open() raster instead!
## it is also still lost when defining transform.. it seems that rasta.plot.show itself contains a transform=transform argument that we can set! voila!
HRtif= rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.tif")
combined_groundtruth_shp = 'C:/Users/Colm The Creator/PycharmProjects/UISB/combined_groundtruth.shp'
combined_groundtruth = gpd.read_file(combined_groundtruth_shp)
orthoHR_lan = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/orthoHR_75mm.lan")
orthoHR_lan = orthoHR_lan.load()
angles_colm_gt = rasterio.open('angles_colm_gt.tif').read()
angles_reduced_gt = rasterio.open('angles_reduced_gt.tif').read()
HR_IR = spy.open_image("C:/Users/Colm The Creator/PycharmProjects/UISB/HR_IR_stacked.lan").load()
HR_IR_lan = HR_IR[:,:,:4]
combined_groundtruth_reduced = combined_groundtruth[47:]
combined_groundtruth_colm = combined_groundtruth[47:73]

groundtruth['coords'] = groundtruth['geometry'].apply(lambda x: x.representative_point().coords[:])
groundtruth['coords'] = [coords[0] for coords in groundtruth['coords']]

# Get list of geometries for all features in vector file
geom = [shapes for shapes in groundtruth.geometry]

# create a numeric unique value for each GT class
groundtruth['u_label'] = groundtruth.groupby(['Label']).ngroup()

## Mask only Rendiermoss as an example
masked_polys = rasta.mask.mask(HRtif,groundtruth[groundtruth['Label'] == 'Rendiermos'].geometry)


groundtruth.plot('Label') ## This takes into account the geometry value...


#############################################################################
## Conversion from .tif to .lan
##  .LAN sensor data type, for reading into SpectralPy, used for spectral analysis in python.
## this is preferably done through terminal vs in python due to GDAL's stability in terminal
#############################################################################
## use this for file conversion in terminal:                 gdal_translate -of LAN orthoHR_75mm.tif orthoHR_75mm.lan
#############################################################################
#######
## use RIO STACK for stacking tiffs in rasterio..
## rio stack orthoHR_75mm.tif ortho25IR.tif -o stacked.tif
#######

## set all masked pixels to class 1
masked_polys[0][0][np.nonzero(masked_polys[0][0])] = 1



#############################################################################
## Spectral Angle Mapper        Rendiermos_example
#############################################################################
groundtruth[['u_label','geometry']]

## create training class loader for SpectralPy
Rendiermos_spy_training_class = spy.create_training_classes(orthoHR_lan,groundtruth['Label'],True)#masked_polys[0][0],True)

## we must first create a CxB array of training class mean spectra, C = training class, B = spectral band
means = np.zeros((len(Rendiermos_spy_training_class), orthoHR_lan.shape[2]), float)
# i = spectral class index, c= spectral class ## c.mask = groundtruth['Label'], c.image = orthoHR_lan.
for (i, c) in enumerate(Rendiermos_spy_training_class):
    ## mean RGB for Rendiermos

    means[i] = c.stats.mean
    # print()
    # spy.save_rgb('spectral_means'+str(i)+'.jpg', c.image)
    # spy.imshow(c.image)

## Calculate Spectral Angle
angles = spy.spectral_angles(orthoHR_lan, means)
#msam = spy.msam(orthoHR_lan, means)
angles[:,:,1].shape
## Here, the minimum angles are the ones closest to the endmember's spectra,
## But this seems to be off a small bit(0.2), possibly because our GT data is not exact?

# plt.imshow(angles.clip(0.15,0.25))
# plt.title('Spectra Angle Map - Rendiermos')
# fig, ax = plt.subplots(figsize=(15, 15))
# rasta.plot.show(HRtif, ax=ax)
# plt.title('orthoHR - Rendiermos')
# groundtruth[groundtruth['Label'] == 'Rendiermos'].plot(ax=ax, facecolor='none', edgecolor='red')
# view = spy.imshow(classes=angles.clip(0.15,0.25))
# plt.title('Spectral Classes - Rendiermos')
# ## spy.imshow acts slightly weird compared to plt,
# plt.pause(100)



##PLT.SUBPLOTS!!!!
##finally i can plot imshows togetehr :)


# plt.subplot(211)
# plt.imshow(angles[:, :, 9])
# plt.subplot(212)
# plt.imshow(angles[:, :, 9])

for i in range(10):
    plt.subplot(4,3,i+1)
    plt.imshow(angles[:,:,i])
plt.show()


# spectral_means = rasterio.open("C:/Users/Colm The Creator/PycharmProjects/UISB/angles_colm_gt.tif")#spectral_means.jpg")
# rasterio.plot.show_hist(spectral_means, bins=250, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)
#
# plt.subplot(321)
# show(spectral_means.read(1))
# plt.subplot(322)
# show(spectral_means.read(2))
# plt.subplot(323)
# show(spectral_means.read(3))
#


#
# ## Trying to see if the mean values used in the GT data is reliable...
# ## will  instead just go through the GT image to see if the GTs look ok :)
# out,_ = rasta.mask.mask(HRtif,groundtruth[groundtruth['Label'] == 'Rendiermos'].geometry)
# rasterio.plot.show_hist(out, bins=250, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3, masked=True)




####################
## ok, so seen as there are a number of species that Stef annotated that aren't in the groundtruth, I'll try to add these to the gt & thereby clean it up also :)
####################

groundtruth_reduced = groundtruth.drop(groundtruth.columns[[0,1,3,4,5,6,7,9,10]],axis=1)
stef_groundtruth_reduced = stef_groundtruth.drop(stef_groundtruth.columns[[0,2,3,4,5]],axis=1)
stef_groundtruth_reduced = stef_groundtruth_reduced.rename(columns={'type_':'Label'})
combined_groundtruth = groundtruth_reduced.append(stef_groundtruth_reduced)
combined_groundtruth['Label'] = combined_groundtruth['Label'].str.lower()

combined_groundtruth = combined_groundtruth.reset_index(drop=True)

## Yes we do have overlapping polygons here, but since we are just using these for masking, it doesn't matter if they overlap..
combined_groundtruth['u_label'] = combined_groundtruth.groupby(combined_groundtruth['Label'].str.lower()).ngroup()
combined_groundtruth.groupby('Label')['index'].nunique()
combined_groundtruth['index'] = combined_groundtruth.index
combined_groundtruth['Label'].replace({'korstmosvetatie':'korstmosvegetatie'})
## removing 1-46, because they suck really bad as GT..
##

####################



## Inspecting GT areas by plotting
out,_ = rasta.mask.mask(HRtif,combined_groundtruth[combined_groundtruth['Label'] == 'Struikheide'].geometry)
show(out)
combined_groundtruth[combined_groundtruth['Label'] == 'Struikheide'].plot('u_label',legend=True)

rasterio.plot.show_hist(out, bins=250, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3, masked=True)



combined_groundtruth.to_file('combined_groundtruth.shp')
combined_groundtruth[0:47].to_file('combined_groundtruth_first_47.shp')
combined_groundtruth[47:73].to_file('combined_groundtruth_colm_47-73.shp')
combined_groundtruth_colm = combined_groundtruth_colm[combined_groundtruth_colm['Label'] != 'strooisel']
combined_groundtruth_colm = gpd.read_file('combined_groundtruth_colm_47-73.shp')

######################
## adding NDVI layer:
NDVI_lan = spy.algorithms.ndvi(HR_IR_lan,0,3)

## again for conversions use gdal in terminal..
spy.save_rgb('NDVI_lan.tif',NDVI_lan)
################
## SAM mapping
################

combined_spy_training_class = spy.create_training_classes(HR_IR_lan,combined_groundtruth_colm['Label'],True)#masked_polys[0][0],True)

## we must first create a CxB array of training class mean spectra, C = training class, B = spectral band
means = np.zeros((len(combined_spy_training_class), HR_IR_lan.shape[2]), float)
# i = spectral class index, c= spectral class ## c.mask = groundtruth['Label'], c.image = orthoHR_lan.
for (i, c) in enumerate(combined_spy_training_class):
    ## mean RGB for Rendiermos

    means[i] = c.stats.mean


angles = spy.spectral_angles(HR_IR_lan, means)

titles_colm = list(combined_spy_training_class.classes.keys())


fig, ax = plt.subplots(2, 3, figsize=(21, 7))
plt.suptitle('combined groundtruth 47:73')
a=0
for i in range(2):
    for j in range(3):
        #plt.subplot(4,4,i+1)
        rasta.plot.show(angles_colm_gt[a], transform=transform,ax=ax[i,j])  # HRtif, ax=ax)
        #plt.title(titles[a],ax=ax[i,j])
        ax[i,j].title.set_text(titles_colm[a])
        #plt.title('orthoHR - Rendiermos')
        combined_groundtruth_colm.plot(ax=ax[i,j], facecolor='none', edgecolor='red')#combined_groundtruth_colm['Label'] == str(titles_colm[a]
        a=a+1

combined_spy_training_class = spy.create_training_classes(HR_IR_lan,combined_groundtruth_reduced['Label'],True)#masked_polys[0][0],True)
titles_reduced = list(combined_spy_training_class.classes.keys())

fig, ax = plt.subplots(4, 4, figsize=(21, 7))
plt.suptitle('combined groundtruth 47:')

a=0
for i in range(4):
    for j in range(4):
        #plt.subplot(4,4,i+1)
        rasta.plot.show(angles_reduced_gt[a], transform=transform,ax=ax[i,j])  # HRtif, ax=ax)
        #plt.title(titles[a],ax=ax[i,j])
        ax[i,j].title.set_text(titles_reduced[a])
        #plt.title('orthoHR - Rendiermos')
        combined_groundtruth_reduced[combined_groundtruth_reduced['Label'] == str(titles_reduced[a])].plot(ax=ax[i,j], facecolor='none', edgecolor='red')
        a=a+1




##########
## Editing Groundtruths:
## pijpenstrootje = purple moor grass,
## bochtige smele = long hair grass.
## Struikheide = heather
## grove den = Scots pine tree
## jeneverbes = juniper berry
## strooisel = litter
## korstmosvegetatie = other moss
## overig = other
##########
## 7 other moss / korstmosvegetatie
## 4 heather / struikheide
## 4 purple moor grass / pijpenstrootje
## 6 wavy hari grass / bochtige smele
## 1 litter / strooisel
## 1 delete
##########
# 47 = korstmosvegetatie #other moss
# 48 = korstmosvegetatie #other moss
# 49 = struikheide #heather
# 50 = pijpenstrootje #purple moor grass
# 51 = korstmosvegetatie #other moss
# 52 = struikheide #heather
# 53 = pijpenstrootje #purple moor grass
# 54 = bochtige smele #wavy hair grass
# 55 = struikheide #heather
# 56 = bochtige smele #wavy hair grass
# 57 = rendiermos
# 58 = delete
# 59 = bochtige smele #wavy hair grass
# 60 = rendiermos
# 61 = rendiermos
# 62 = korstmosvegetatie #other moss
# 63 = korstmosvegetatie #other moss
# 64 = strooisel #litter
# 65 = korstmosvegetatie #other moss
# 66 = bochtige smele #wavy hair grass
# 67 = pijpenstrootje #purple moor grass
# 68 = pijpenstrootje #purple moor grass
# 69 = bochtige smele #wavy hair grass
# 70 = korstmosvegetatie #other moss
# 71 = struikheide #heather
# 72 = rendiermos
# 73 = bochtige smele #wavy hair moss
# 74 = bochtige smele
# 75 = pijpenstrootje
# 76 = struikheide
# 77 = bochtige smele
# 78 = delete
# 79 = bochtige smele




# combined_groundtruth.loc[[47,48,51,62,63,65,70],'Label'] = 'korstmosvegetatie'
# combined_groundtruth.loc[[49,52,55,71,76],'Label'] = 'struikheide'
# combined_groundtruth.loc[[50,53,67,68,75],'Label'] = 'pijpenstrootje'
# combined_groundtruth.loc[[54,56,59,66,69,73,74,77,79],'Label'] = 'bochtige smele'
# combined_groundtruth.loc[[57,60,61,72],'Label'] = 'rendiermos'
# combined_groundtruth.loc[[58,78],'Label'] = 'delete'
# combined_groundtruth.loc[[64],'Label'] = 'strooisel'





