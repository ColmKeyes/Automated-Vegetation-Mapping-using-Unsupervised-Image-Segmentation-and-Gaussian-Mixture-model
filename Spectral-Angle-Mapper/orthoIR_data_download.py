
from owslib.wms import WebMapService
import rasterio.mask
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint
import rasterio.plot
import fiona










######GET INFRARED IMAGE 25CM RES######
#same procedure as for the RGB image
Url_to_infrared= 'https://service.pdok.nl/hwh/luchtfotocir/wms/v1_0?request=GetCapabilities&service=wms'

wms = WebMapService(Url_to_infrared)

# wms['Actueel_orthoHR'].title
# wms['Actueel_orthoHR'].queryable
# wms['Actueel_orthoHR'].opaque
#wms['Actueel_orthoHR'].crsOptions
#ms['Actueel_ortho25IR'].styles   #nostyles
#wms.getOperationByName('GetMap').formatOptions
#wms['Actueel_orthoHR'].boundingBox   #no bounding
#[op.name for op in wms.operations]  #getcapabilities, getmap, getfeatureinfo

# the infrared image has a resolution of 25cm. Using the same xsize and ysize as the RGB image
# resamples the image to the same resolution of 7.5cm.

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

imgIR = wms.getmap( layers=['Actueel_ortho25IR'],
                  srs='EPSG:28992',
                  bbox=(163920, 367450, 165370, 368700),
                  size=(xsize, ysize),
                  format='image/jpeg',
                  transparent=False)


out = open('ortho25IR.jpeg', 'wb')
out.write(imgIR.read())
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

#open and georeference the Infrared image and saving as a tif file.
IRtif= rasterio.open('ortho25IR.jpeg')
IRtif= IRtif.read([1,2,3])

with rasterio.open('ortho25IR.tif',
                   'w',
                   driver='GTiff',
                   count=IRtif.shape[0],
                   height=IRtif.shape[1],
                   width=IRtif.shape[2],
                   dtype=IRtif.dtype,
                   crs=crs,
                   transform=transform,

                   ) as dst:
    dst.write(IRtif)

    # open study area (have to upload a study area shapefile for this)

    with fiona.open("data/Study_area/Study_area.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # transform the studyarea to 2D cause its in 3D
    geoms2d = [
        {
            "type": g["type"],
            "coordinates": [[xyz[0:2] for xyz in p] for p in g["coordinates"]],
        }
        for g in shapes
    ]


# for IR image
with rasterio.open("ortho25IR.tif") as src:
    out_image, out_transform = rasterio.mask.mask(src, geoms2d, crop=True)
    out_meta = src.meta

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open("ortho25IRmasked.tif", "w", **out_meta) as dest:
    dest.write(out_image)

