"""
@Time    : 25/07/2022 10:13
@Author  : Colm Keyes
@Email   : keyesco@tcd.ie
@File    : orthoHR_data_download.py
@License : MIT
"""

#############################################################################
## orthoHR data download - Georeference & Save tiff
#############################################################################


######GET RGB IMAGE 7.5CM RES######

from owslib.wms import WebMapService
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import rasterio.plot



# paste here the url to the web map service
Url_To_RGB = 'https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?request=GetCapabilities&service=wms'
wms = WebMapService(Url_To_RGB)

# list of all maps in the web map service
list(wms.contents)
# check the metadata of the layer
print(wms['Actueel_orthoHR'].abstract)
print(wms.getOperationByName('GetMap').formatOptions)


# a boundingbox was drawn using the website: bboxfinder. To get the original resolution, the length and the width of the boundingbox were divided by the amount of pixels per meter.
xsize = (16537 - 16392) * (100 / 7.5)
ysize = (36870 - 36745) * (100 / 7.5)
print(xsize)
print(ysize)

# coordinates of the boundingbox
x1 = 163920
y1 = 368700
x2 = 165370
y2 = 367450

# import the map from the web map service. Unfortunately, for this map only a non georeferenced jpeg image was available.
img = wms.getmap(layers=['Actueel_orthoHR'],
                 srs='EPSG:28992',
                 bbox=(x1, y2, x2, y1),
                 size=(xsize, ysize),
                 format='image/png',  # 'image/jpeg',
                 transparent=False)

# save the image
out = open('orthoHR_75mm.jpeg', 'wb')
out.write(img.read())
out.close()

tl = GroundControlPoint(0, 0, x1, y1)
bl = GroundControlPoint(1666.6666666666667, 0, x1, y2)  # 1666, 0, x1, y2)
br = GroundControlPoint(1666.6666666666667,  1933.3333333333335, x2, y2)
tr = GroundControlPoint(0,  1933.3333333333335, x2, y1)
gcps = [tl, bl, br, tr]

## Affine transform is a simple total transform which will transform each point on a 2d axis in each direction, which is given by a 3x3 array for left right lower left lower right etc.
## our transform will be completed using ground points to transform the points to an exact pixel position.

# transform to GTIFF
transform = from_gcps(gcps)
crs = rasterio.crs.CRS.from_epsg(28992)  # 'epsg:28992'

## Open jpeg just to get it's shapes etc, before using these as necessary extents to form the tif file..
HRtif_metas = rasterio.open('orthoHR_75mm.jpeg')

## This line below is needed to read the 3 bands into memory.. otherwise we will have just [w,h] instead of [layer,w,h]
HRtif_metas_read = HRtif_metas.read([1, 2, 3])

## Save image to tif with metadata
with rasterio.open('orthoHR_75mm.tif',
              'w',
              driver='GTiff',
              count=HRtif_metas_read.shape[0],
              height=HRtif_metas_read.shape[1],
              width=HRtif_metas_read.shape[2],
              dtype=HRtif_metas_read.dtype,
              crs=crs,
              transform=transform) as HRtif_1:
                    HRtif_1.write(HRtif_metas_read)
