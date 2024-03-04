# -*- coding: utf-8 -*-
import ee
import google
import os, sys, io
import geemap

from geemap import ml, ee_to_pandas
from pprint import pprint
import pandas as pd
from sklearn import ensemble

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# In[6]:


#ee_key_file = '../floating-weed-manager_02-fdb371c09333.json' # "../floating-weed-manager-6366298ed5f9.json"
#service_account = 'jhe-fwm-02@floating-weed-manager.iam.gserviceaccount.com' # 'jhe-fwm-01@floating-weed-manager.iam.gserviceaccount.com'

ee_key_file = "../floating-weed-manager-69cbf037de43.json"
service_account = 'floating-weed-manager@appspot.gserviceaccount.com'

credentials = ee.ServiceAccountCredentials(service_account, ee_key_file)
ee.Initialize(credentials)
from google.oauth2 import service_account
sheet_service = build('sheets', 'v4', credentials=service_account.Credentials.from_service_account_file(ee_key_file))
drive_service = build('drive', 'v3', credentials = service_account.Credentials.from_service_account_file(ee_key_file))

"""# Functions"""

# =========================================== Cloud mask Functions =========================================== #

# S2cloudless ---------------------------------------------------------------------------------------------------
#*
 # This set of functions builds an S2 and S2 cloud probability collection, masks cloud and cloud shadow
 # author - jdbcode https:#developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
 #
 # FUNCTIONS TO CALL IN SCRIPT:
 # 1. get_s2_col - Builds S2 collection
 # 2. add_cld_shdw_mask - Adds cloud and cloud shadow bands
 # 3. apply_cld_shdw_mask - Applies cloud and cloud shadow mask
 #
 # SETTINGS:
 # param {number} CLOUD_FILTER - Maximum image cloud cover percent allowed in image collection
 # param {number} CLD_PRB_THRESH - Cloud probability (%); values greater than are considered cloud
 # param {number} NIR_DRK_THRESH - Near-infrared reflectance; values less than are considered potential cloud shadow
 # param {number} CLD_PRJ_DIST - Maximum distance (km) to search for cloud shadows from cloud edges
 # param {number} BUFFER - Distance (m) to dilate the edge of cloud-identified objects
 #

# SETTINGS:
CLOUD_FILTER = 60
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 2
BUFFER = 100

#*
 # This function builds an S2 collection and joins with S2 cloud probability collection
 # author - jdbcode
 # param {geometry} aoi - area of interest
 # param {string} start_date - start date of image collection
 # param {string} end_date - end date of image collection
 # return {imageCollection} - S2 Image Collection with S2 cloud probability for chosen location and date range.
 #
def get_s2_col(aoi, start_date, end_date):
  # filter S2 collection
   s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date))

  # filter s2cloudless collection
   s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date))

  # join collections by the 'system:index' property.
   return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
     'primary': s2_sr_col,
     'secondary': s2_cloudless_col,
     'condition': ee.Filter.equals(**{
       'leftField': 'system:index',
       'rightField': 'system:index'
        })
    }))

#*
 # This function adds a cloud mask from S2 cloud probability layer
 # author - jdbcode
 # param {image} img - S2 Image with cloud probability layer
 # return {image} - S2 Image with cloud probability layer and cloud mask as bands
 #
def add_cloud_bands(img):
  # get s2cloudless probability
   cld_prb = ee.Image(img.get('s2cloudless')).select('probability')

  # set clouds as greater than CLD_PRB_THRESH setting
   is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds')

  # add the cloud probability layer and cloud mask as image bands.
   return img.addBands(ee.Image([cld_prb, is_cloud]))

#*
 # This function calculates dark pixels and cloud projection to identify cloud shadow
 # #Edit# Removed code that prevented identification of dark pixels over water.
 # author - jdbcode
 # param {image} img - S2 Image with cloud probability layer
 # return {image} - S2 Image with dark pixels, cloud projection and identified shadows added as bands.
 #
def add_shadow_bands(img):
  # identify water pixels from the SCL band.
   not_water = img.select('SCL').neq(6)

  # identify dark NIR pixels that are not water
   SR_BAND_SCALE = 1e4
   dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE) \
                    .rename('dark_pixels')

  # determine the direction to project cloud shadow from clouds (assumes UTM projection).
   shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))

  # project shadows from clouds for the distance specified by the CLD_PRJ_DIST setting.
   cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10) \
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100}) \
        .select('distance') \
        .mask() \
        .rename('cloud_transform'))

  # identify the intersection of dark pixels with cloud shadow projection.
   shadows = cld_proj.multiply(dark_pixels).rename('shadows')

   # add dark pixels, cloud projection, and identified shadows as image bands.
   return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

#*
 # This function assembles cloud and cloud shadow components and produces final mask
 # author - jdbcode
 # param {image} img - S2 Image with dark pixels, cloud projection, and identified shadows bands
 # return {image} - S2 Image with added cloud and cloud shadow bands
 #
def add_cld_shdw_mask(img):
  # add cloud component bands.
   img_cloud = add_cloud_bands(img)

  # add cloud shadow component bands.
   img_cloud_shadow = add_shadow_bands(img_cloud)

  # combine cloud and shadow mask, set cloud and shadow as value 1, else 0.
   is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)

    # remove small cloud-shadow patches and dilate remaining pixels by BUFFER setting.
    # 20 m scale is for speed, and assumes clouds don't require 10 m precision.
   is_cld_shdw = (is_cld_shdw.focal_min(2).focal_max(BUFFER*2/20) \
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20}) \
        .rename('cloudmask'))

    # add the final cloud-shadow mask to the image.
   return img_cloud_shadow.addBands(is_cld_shdw)

#*
 # This function applies cloud mask to each image in the collection
 # author - jdbcode
 # param {image} img - S2 Image with cloud-shadow band
 # return {image} - S2 Image with clouds masked
 #
def apply_cld_shdw_mask(img):
  # subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
   not_cld_shdw = img.select('cloudmask').Not()

  # subset reflectance bands and update their masks, return the result.
   return img.select('B.*').updateMask(not_cld_shdw)

# =========================================== Water mask Functions =========================================== #

#*
 # This function adds a water band by thresholding a selected spectral index
 # author - Henry Thompson
 # param {image} img - S2 Image with renamed bands {@link selectrenameBands} and added indices {@link addIndices}
 # return {image} - S2 Image with added water band
 #
def addWater(img):
  # set water threshold
  threshold = 0

  # choose water index
  index = 'AWEI'
          # 'MNDWI'

  # select water greater than threhold
  waterMask = img.select(index).gte(threshold).rename('Water')

  # add water band
  img = img.addBands(waterMask)

  return img

# =========================================== Pre-processing Functions =========================================== #

#*
 # This function selects Sentinel-2 bands and renames them
 # author - Henry Thompson
 # param {image} image - Sentinel-2 Image
 # return {image} - Image with selected bands
 #
def selectrenameBands(image):
  image = image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'cloudmask'],
                     ['B1', 'B', 'G', 'R', 'RE1', 'RE2', 'RE3', 'NIR', 'RE4', 'SWIR', 'SWIR2', 'cloudmask'])
  return image
#*
 # This function calculates spectral indices from Sentinel-2 and adds them as bands
 # author - Henry Thompson
 # param {image} image - Sentinel-2 Image
 # return {image} - Image with indices added as bands
 #
def addIndices(image):
  #normalised difference moisture index
  ndmi =  image.normalizedDifference(['NIR', 'SWIR']).rename('NDMI').set('date', image.get('date'))
  #automated water extraction index
  awei = image.expression('B+2.5*G-1.5*(NIR+SWIR)-0.25*SWIR2',{
      'B' : image.select('B'), 'G' : image.select('G'),
      'NIR' : image.select('NIR'), 'SWIR' : image.select('SWIR'),
      'SWIR2' : image.select('SWIR2')
  }).rename('AWEI').set('date', image.get('date'))
  aweinsh = image.expression('4*(G - SWIR)-(0.25*NIR+2.75*SWIR2)',{
      'G' : image.select('G'),
      'NIR' : image.select('NIR'), 'SWIR' : image.select('SWIR'),
      'SWIR2' : image.select('SWIR2')
  }).rename('AWEInsh').set('date', image.get('date'))
  return image.addBands([ndmi,awei,aweinsh])
#*
 # This function calculates "entropy" texture measure and adds as a band
 # author - Henry Thompson
 # param {image} image - Sentinel-2 Image
 # return {image} - Image with "entropy" and added
 #
def addTexture(image):
  # select band for texture analysis
  texture = image.select(['NIR'], ['entropy'])

  # define a neighborhood with a kernel.
  square = ee.Kernel.square(radius=9)

  # compute entropy and add band.
  entropy = texture.entropy(square)

  return image.addBands([entropy])

#*
 # This function calculates selected GLCM textures on every and in Sentinel-2 image
 # author - Henry Thompson
 # param {image} image - Sentinel-2 Image
 # return {image} - Image with GLCM textures added as bands
 #
def addGLCM(img):
  # select window size
  textures = img.glcmTexture(size = 3) \
  .select(['.*_savg', '.*_var',
          '.*_contrast', '.*_diss',
          '.*_ent', '.*_asm',
          '.*_corr'])

  return img.addBands(textures)

#*
 # This function adds a day of year (DOY) band to an image
 # author - Henry Thompson
 # param {image} image - Image
 # return {image} - Image with DOY added as band
 #
def addDOY(img):
  # get doy
  doy = img.date().getRelative('day', 'year')

  # create doy band
  doyBand = ee.Image.constant(doy).uint16().rename('DOY_image')

  return img.addBands(doyBand)


#*
 # This function mosaics images from the same day
 # author - Henry Thompson
 # param {imageCollection} imcol - Sentinel-2 Image Collection
 # return {imageCollection} - Image Collection of mosaiced images, image properties carried over from the first image in a mosaic
 #
def mosaicByDate(imcol):
  # image collection to list
  imlist = imcol.toList(imcol.size())

  # function to format date
  def format_date(im):
    return ee.Image(im).date().format("YYYY-MM-dd")

  # create list of unique dates
  unique_dates = imlist.map(format_date).distinct()

  # function to map over dates list and mosaic images from same day
  def mosaic_images(d):
      # filter images by date
      d = ee.Date(d)
      imcol_d = imcol.filterDate(d, d.advance(1, "day"))

      # get chosen properties from first image of mosaic
      product_id = imcol_d.first().get("PRODUCT_ID")

      # mosaic images
      im = imcol_d \
        .mosaic()

      # set mosaicked image properties
      return im.set(
          "system:time_start", d.millis(),
          "system:id", d.format("YYYY-MM-dd"),
          "PRODUCT_ID", product_id)

  mosaic_imlist = unique_dates.map(mosaic_images)

  return ee.ImageCollection(mosaic_imlist)

# ================================================= Classify ================================================= #

# Function to classify each image in a collection
def classify_collection(image):
  # NDMI thresholds
  ndmi_lower = 0.3 
  ndmi_upper = 0.63 

  # Select bands
  ndmi = image.select('NDMI')
  binary_water = image.select('Water')

  # Classify
  not_water = binary_water.Not()
  binary_other = (ndmi.lt(ndmi_lower).Or(ndmi.gte(ndmi_upper))).multiply(not_water)
  binary_floating = (ndmi.gte(ndmi_lower).And(ndmi.lt(ndmi_upper))).multiply(not_water)
  binary_cloud = image.select('cloudmask')

  # Combine into single band
  classified = (binary_water.add(binary_other.multiply(2)).add(binary_floating.multiply(3))).multiply(binary_cloud.Not())
  return classified.set('system:time_start', image.get('system:time_start'))

from datetime import date, timedelta, datetime

def set_location_time(argv):
  today = date.today()
  tomorrow = today + timedelta(days=1)
  roi_name = 'LakeO'
  start_dt_str = today.isoformat()
  end_dt_str = tomorrow.isoformat()
  n = len(argv)
  print(f'number arg: {n}')
  if n == 1:
    roi_name = argv[0]
  elif n == 2:
    roi_name = argv[0]
    start_dt_str = argv[1]
  elif n == 3:
    roi_name = argv[0]
    start_dt_str = argv[1]
    end_dt_str = argv[2]
  else:
    pass
  return roi_name, start_dt_str, end_dt_str

def get_argvs():
  with open('resume.csv') as f:
    argv = [s.strip() for s in f.readline().split(',')]
  zt = set_location_time(argv)
  print(zt)
  return zt

def save_check_point(roi_name, dt_str):
  dt = datetime.strptime(dt_str, '%Y-%m-%d') + timedelta(days=1)
  with open('resume.csv', 'w') as f:
    f.write(f'{roi_name},{dt.date().isoformat()}')

roi_name, start_dt_str, end_dt_str = get_argvs() # YYYY-MM-DD
print(roi_name)

# Study Site (lake boundary) # TODO: USACE_{roi_name}_Polygons
roi = ee.FeatureCollection(f"projects/floating-weed-manager/assets/Polygons/{roi_name}_S2_Channels_Landmask").geometry()

# Study Period
start, end = ee.Date(start_dt_str), ee.Date(end_dt_str)

"""# Pre-process Sentinel-2 Imagery"""

# Get S2 Collection
S2_Col = get_s2_col(roi, start, end) \
              .map(add_cld_shdw_mask) \
              .map(selectrenameBands) \
              .map(addIndices) \
              .map(addWater) \
              .filter(ee.Filter.eq('GENERAL_QUALITY', 'PASSED')) # quality check 

if S2_Col.size().getInfo()==0: sys.exit()

# Mosaic tiles from same day and add a day of year band
S2_Col = mosaicByDate(S2_Col).map(addDOY)

# Print number of mosaiced images
if S2_Col.size().getInfo()==0: sys.exit()

"""## Apply Function to classify each image
1. Classify image - Other = 2 Floating = 3
2. Unmask the cloud mask from NA to 0
3. Mask water and unmask as 1

Final classes...
NA (i.e. missing data) = nodata
Cloud = 0
Water = 1
Other = 2
Floating = 3
"""

# Visualise raw imagery
def visualise_rgb(img):
    trueColour = {"bands": ["R", "G", "B"],"min": 0,"max": 3000}
    img_viz = img.visualize(**trueColour)
    return img_viz

raw_collection = S2_Col.map(visualise_rgb)
# raw_collection = S2_Col.select(['B1', 'B', 'G', 'R', 'SWIR'])
# pprint(raw_collection.first().getInfo())

# classify image with trained rf classifier
classified_collection = S2_Col.map(classify_collection)

pprint(classified_collection.first().getInfo())

"""## Export"""

## Batch export

# print out the number of images in the ImageCollection
count = classified_collection.size().getInfo()
print("Count: ", count)

# export the ImageCollection to Google Drive
image_list = classified_collection.toList(count)
raw_list = raw_collection.toList(count)

dt_list = []
for i in range(0, count):
    image = ee.Image(image_list.get(i)).clip(roi).toFloat()
    raw_image = ee.Image(raw_list.get(i)).clip(roi).toFloat()
    this_time = str(ee.Date(ee.Image(image_list.get(i)).date()).format('YYYY-MM-dd').getInfo())
    print(f'{i}: {this_time}')
    dt_list.append(this_time)
    task1 = ee.batch.Export.image.toDrive(
      image=image,
      description="_".join(['S2', roi_name, this_time]),
      region=roi, # .geometry(),
      folder='classified_maps',
      scale=20,
      crs='EPSG:3857'
    )
    task2 = ee.batch.Export.image.toDrive(
      image=raw_image,
      description="_".join(['S2_Raw', roi_name, this_time]),
      region=roi, # .geometry(),
      folder = 'raw_images',
      scale=20,
      crs='EPSG:3857',
      fileFormat='GeoTIFF',
      formatOptions={
        "cloudOptimized": True
      }
    )
    task1.start()
    task2.start()
    print(task1.status())
    print(task2.status())
    
dt_list.sort()
save_check_point(roi_name, dt_list[-1])

"""# Rolling weekly composite with S2"""
# "User memory limit exceeded"
# Is this going to crash everything? Better to create a composite of images from the archive instead?

# pprint(S2_Col.first().getInfo())

def weekly_composites(img):
    # filter the previous 7 days of imagery
    img_date = ee.Date(img.get('system:time_start'))
    start_date = img_date.advance(-7, "day")
    
    # filter week's worth of images
    weekly_col = get_s2_col(roi, start_date, img_date) \
                  .map(add_cld_shdw_mask) \
                  .map(selectrenameBands) \
                  .map(addIndices) \
                  .map(addWater) \
                  .filter(ee.Filter.eq('GENERAL_QUALITY', 'PASSED')) # quality check 
   
    #classify imagery
    weekly_fw = weekly_col.map(classify_collection)
    
    # function to create FW binary
    def binary_fw(img):
        fw = img.eq(3)
        return fw
    
    # create fw binary
    weekly_fw_mean = weekly_fw.map(binary_fw).mean()
    
    # add properties (start and end date)
    
    return weekly_fw_mean.set({'system:time_start': ee.Date(start_date).millis(),
                              'system:time_end': ee.Date(img_date).millis()})

classified_weekly = S2_Col.map(weekly_composites)

pprint(classified_weekly.first().getInfo())


"""## Export"""

## Batch export

# print out the number of images in the ImageCollection
count = classified_weekly.size().getInfo()
print("Count: ", count)

# export the ImageCollection to Google Drive
image_list = classified_weekly.toList(count)

dt_list = []
for i in range(0, count):
    image = ee.Image(image_list.get(i)).clip(roi).toFloat()
    this_time = str(ee.Date(ee.Image(image_list.get(i)).date()).format('YYYY-MM-dd').getInfo())
    task1 = ee.batch.Export.image.toDrive(
      image=image,
      description="_".join(['S2_weekly', roi_name, this_time]),
      region=roi, # .geometry(),
      folder='classified_maps',
      scale=20,
      crs='EPSG:3857'
    )
    task1.start()
    print(task1.status())  
    
"""# Timeseries by USACE Management Zone

## Apply function
"""

# Load USACE management zones
usace_zones = ee.FeatureCollection(f"projects/floating-weed-manager/assets/Polygons/USACE_{roi_name}_Polygons") # .geometry()

# Merge with whole lake polygon 
lake_ft = ee.Feature(roi, {'id': None, 'Name': roi_name})

# Merge the new feature into the existing feature collection
updated_fc = usace_zones.merge(ee.FeatureCollection([lake_ft]))

# Print the updated feature collection
# print(updated_fc.getInfo())
usace_zones_with_lake = usace_zones

# Function to calculate total change per change image in a single ROI
class_vals = [0,1,2,3]
class_names = ['Cloud','Water','Other','Floating']

def class_calc(img):
  # get properties
  img_date = img.get('system:time_start') # .format('YYYY-MM-dd')

  # Remap to change class name and get area (km2)
  class_count = img.eq(class_vals).rename(class_names)
  class_area = class_count.multiply(ee.Image.pixelArea()); #m2

  # add pixel area band as constant
  class_area = class_area.addBands(ee.Image.pixelArea().rename('pixel_area'))

  # Reduce regions to get total area in each ROI
  total_area = class_area.reduceRegions(**{
    'reducer': ee.Reducer.sum(),
    'collection': usace_zones_with_lake,
    'scale': 20
  })

  def create_output(ft):
    # calculate percentages
    water_perc = ee.Number(ft.get('Water')).divide(ee.Number(ft.get('pixel_area'))).multiply(100)
    cloud_perc = ee.Number(ft.get('Cloud')).divide(ee.Number(ft.get('pixel_area'))).multiply(100)
    other_perc = ee.Number(ft.get('Other')).divide(ee.Number(ft.get('pixel_area'))).multiply(100)
    floating_perc = ee.Number(ft.get('Floating')).divide(ee.Number(ft.get('pixel_area'))).multiply(100)

    # Select properties and output to new feature
    return ee.Feature(None, {'id': ft.get('id'),
                            'name': ft.get('Name'),
                            'water_m2': ft.get('Water'),
                            'cloud_m2': ft.get('Cloud'),
                            'other_m2': ft.get('Other'),
                            'floating_m2': ft.get('Floating'),
                            'roi_m2': ft.get('pixel_area'),
                            'water_percent': water_perc,
                            'cloud_percent': cloud_perc,
                            'other_percent': other_perc,
                            'floating_percent': floating_perc,
                            'system_time_start': img_date,
                            'date' : ee.Date(img_date).format('YYYY-MM-dd')}) #create featurecollection

  output_ftC = total_area.map(create_output)

  return(output_ftC)

# Map over imagecollection and flatten output to featurecollection
class_ftC = ee.FeatureCollection(classified_collection.map(class_calc)).flatten()

# print(class_ftC.limit(3))

# spreadsheetId = '1Zo7eriL87SxLPMQ8UtwyX-P3T3qMrlJavJpQ6_xomEU' # S2_Timeseries_USACEZones_2022
spreadsheetId = '1Cs34tqkJ2qk8_JDIIFaV89iQ8PrqsjhXr3ImyIGy5rw' # S2_Timeseries_USACEZones_2023

def get_last_row_color(service, spreadsheetId):
    result = service.spreadsheets().get(spreadsheetId=spreadsheetId, ranges="Sheet1!A:A", includeGridData=True).execute()
    sheetId = result['sheets'][0]['properties']['sheetId']
    rowCount = result['sheets'][0]['properties']['gridProperties']['rowCount']
    sheet_data = result['sheets'][0]['data'][0]
    last_row_properties = sheet_data['rowData'][-1]['values'][0]['userEnteredFormat']
    bg = last_row_properties.get('backgroundColor', {}).copy()
    return sheetId, rowCount, bg

sheetId, rowCount, bg = get_last_row_color(sheet_service, spreadsheetId) # [201,218,248], [0.78823529, 0.85490196, 0.97254902]
bg['blue'] = (bg['blue'] + 0.5) % 1

# Select columns
selectors = ['cloud_m2','cloud_percent','floating_m2','floating_percent','id',
             'name','other_m2','other_percent','roi_m2','system_time_start','water_m2',
             'water_percent','date']
rows = ee_to_pandas(class_ftC, col_names=selectors).values.tolist()

# property_list = []
# for feature in class_ftC.select(selectors).getInfo()['features']:
#     properties = feature['properties']
#     property_list.append(list(properties.values()))
# # print(property_list)
resource = {
  "majorDimension": "ROWS",
  "values": rows
}

range = "Sheet1!A:A";
sheet_service.spreadsheets().values().append(
  spreadsheetId=spreadsheetId,
  range=range,
  body=resource,
  valueInputOption="USER_ENTERED"
).execute()
print('append timeserie rows')

request_body = {"requests": [{
    "repeatCell": {
        "range": {
            "sheetId": sheetId,
            "startRowIndex": rowCount,
            "endRowIndex": rowCount + len(rows),
            "startColumnIndex": 0,
        },
        "cell": {
            "userEnteredFormat": {
                "backgroundColor": bg
            }
        },
        "fields": "userEnteredFormat.backgroundColor"
    }
}]}

request = sheet_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheetId, body=request_body)
res = request.execute()

# df = ee_to_pandas(class_ftC, col_names=selectors)
# file_id = '1DxFjgq40O1wKo8EPqx2BUDY5ixJJF5ho' # S2
# csv_str = df.to_csv(index=None) # ,header=False)
# media = MediaIoBaseUpload(io.BytesIO(csv_str.encode()), mimetype='text/csv', resumable=True)
# updated_file = drive_service.files().update(fileId=file_id, media_body=media).execute()
# print(updated_file.get("id"))

# # Export featurecollection
# task = ee.batch.Export.table.toDrive(
#   collection = class_ftC,
#   selectors = selectors,
#   description = "_".join([
#     'S2', 'Timeseries', 'USACEZones',
#     str(start.format('YYYY-MM-dd').getInfo()),
#     str(end.format('YYYY-MM-dd').getInfo())
#   ]),
#   folder = 'timeseries',
#   fileFormat = 'CSV'
# )
# task.start()
# print(task.status())