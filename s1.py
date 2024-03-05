#!/usr/bin/env python

# -*- coding: utf-8 -*-
import ee
import google
import os, sys, io

from geemap import ml, ee_to_pandas  # note new module within geemap
from pprint import pprint
import math

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

#ee_key_file = "../floating-weed-manager-6366298ed5f9.json"    # 01
#ee_key_file = "../floating-weed-manager_02-fdb371c09333.json" # 02
#service_account = 'jhe-fwm-02@floating-weed-manager.iam.gserviceaccount.com' # 'jhe-fwm-01@floating-weed-manager.iam.gserviceaccount.com'

ee_key_file = "../floating-weed-manager-69cbf037de43.json"
service_account = 'floating-weed-manager@appspot.gserviceaccount.com'

credentials = ee.ServiceAccountCredentials(service_account, ee_key_file)
ee.Initialize(credentials)
from google.oauth2 import service_account
sheet_service = build('sheets', 'v4', credentials=service_account.Credentials.from_service_account_file(ee_key_file))
drive_service = build('drive', 'v3', credentials = service_account.Credentials.from_service_account_file(ee_key_file))

"""# Functions"""

# =========================================== Pre-processing Functions =========================================== #

#*
 # This function smooths an image by focal median
 # author - Henry Thompson
 # param {image} img - Sentinel-1 Image
 # return {image} - Image with smoothed bands adwded
 #
def smoothFocalMedian(img):
  # select VV and VH polarisations
  vv = img.select('VV')
  vh = img.select('VH')
  # apply focal median filter
  vv_smoothed = vv.focal_median(50, 'circle', 'meters').rename('VV_smooth') #(50, 'circle', 'meters') #(5, 'circle', 'pixels')
  vh_smoothed = vh.focal_median(50, 'circle', 'meters').rename('VH_smooth')
  # add smoothed bands to image
  return img.addBands(vv_smoothed).addBands(vh_smoothed) # Return smoothed image

#*
# This function smooths an image by Perona-Malik (anisotropic diffusion) convolution
# author - https://www.mdpi.com/2072-4292/12/15/2469/htm
# param {image} img - Sentinel-1 Image
# param iter: Number of interations to apply filter
# param K: kernal size
# param opt_method: choose method 1 (default) or 2, DETAILS
# return {image} - single band ee.Image in natural units
#
# Define the PeronaMalik function 
def smoothPeronaMalik(I, iter=10, K=3, opt_method=1):
    # Currently just accepts a single image. Use the smoothImage helper function to run different bands
    iter = iter ##or 10  # Default value for iter
    K = K ##or 3
    method = opt_method ##or 1

    # Define kernels
    dxW = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    dxE = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    dyN = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    dyS = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    lambda_ = 0.2
    k1 = ee.Image(-1.0 / K)
    k2 = ee.Image(K).multiply(ee.Image(K))

    for i in range(iter):
        dI_W = I.convolve(dxW)
        dI_E = I.convolve(dxE)
        dI_N = I.convolve(dyN)
        dI_S = I.convolve(dyS)

        if method == 1:
            cW = dI_W.multiply(dI_W).multiply(k1).exp()
            cE = dI_E.multiply(dI_E).multiply(k1).exp()
            cN = dI_N.multiply(dI_N).multiply(k1).exp()
            cS = dI_S.multiply(dI_S).multiply(k1).exp()
            I = I.add(ee.Image(lambda_).multiply(cN.multiply(dI_N).add(cS.multiply(dI_S)).add(cE.multiply(dI_E)).add(cW.multiply(dI_W))))
        elif method == 2:
            cW = ee.Image(1.0).divide(ee.Image(1.0).add(dI_W.multiply(dI_W).divide(k2)))
            cE = ee.Image(1.0).divide(ee.Image(1.0).add(dI_E.multiply(dI_E).divide(k2)))
            cN = ee.Image(1.0).divide(ee.Image(1.0).add(dI_N.multiply(dI_N).divide(k2)))
            cS = ee.Image(1.0).divide(ee.Image(1.0).add(dI_S.multiply(dI_S).divide(k2)))
            I = I.add(ee.Image(lambda_).multiply(cN.multiply(dI_N).add(cS.multiply(dI_S)).add(cE.multiply(dI_E)).add(cW.multiply(dI_W))))

    return I

# Helper function to smooth both VV and VH polarisations using chosen methods
def smoothImage(image):
    smoothed_VH = smoothPeronaMalik(image.select('VH'), iter=10, K=3, opt_method=1)
    smoothed_VV = smoothPeronaMalik(image.select('VV'), iter=10, K=3, opt_method=1)
    return image.addBands(smoothed_VH.rename('VH_smooth')).addBands(smoothed_VV.rename('VV_smooth'))

# =========================================== Other Functions =========================================== #
#*
 # This function mosaics images from the same day
 # author - Henry Thompson
 # param {imageCollection} imcol - Sentinel-1 Image Collection
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

# =========================================== Water mask Functions =========================================== #

#*
 # This function adds a water band by thresholding smoothed VH and/or VV bands
 # author - Henry Thompson
 # param {image} img - S1 Image with smoothed bands {@link smoothFocalMedian}
 # return {image} - S1 Image with added water band - "Water_thresh"
 #
def addWater(img):
  vv = img.select('VV_smooth')
  vh = img.select('VH_smooth')

  # set water threshold
  thresh = -23

  # threshold water
  water = vh.lte(thresh) \
              .rename('Water')

  return img.addBands(water)  #Return image with added classified water band


#*
 # This function adds an indicence angle corrected water band by thresholding smoothed VH and/or VV bands
 # author - Henry Thompson
 # param {image} img - S1 Image with smoothed bands {@link smoothFocalMedian}
 # return {image} - S1 Image with added corrected water band - "Water_SCC"
 #
def addWaterCorrected(img):
  # get bands and angle
  vv = img.select('VV_smooth')
  vh = img.select('VH_smooth')
  angle = img.select('angle')

  # convert angle to radians
  sin_angle = angle.divide(180).multiply(math.pi).sin()

  # correct VV and VH bands using angle (SCC method)
  vv_corrected = vv.multiply((sin_angle.cos()).pow(2)).rename('VV_corrected')
  vh_corrected = vh.multiply((sin_angle.cos()).pow(2)).rename('VH_corrected')

  # threshold water
  water = vh_corrected.lte(-14) \
          .rename('Water')

  return img.addBands([water, vh_corrected])  #Return image with added classified water band

# ============================================== Classify and Change ============================================== #

def classify_macrophytes(image):
  classified = image.select('Water').Not()
  return classified.set('system:time_start', image.get('system:time_start'))


def create_change_collection(img):
    # get current image index
    index = macrophyte_list.indexOf(img)
    img = ee.Image(img)
    # get previous image index
    previousIndex = ee.Algorithms.If(index.eq(0), index, index.subtract(1))
    previousImage = ee.Image(macrophyte_list.get(previousIndex))
    # calculate change:
      # 0: NA
      # 1: water -> land
      # 2: land -> water
      # 3: land -> land
      # 4: water -> water
    change = ee.Image(previousImage.select('Water').multiply(10) \
                                      .add(img.select('Water')) \
                                      .remap([10, 1, 0, 11], [1, 2, 3, 4]) \
                                      .rename('change') \
                                      .set({'system:time_start': previousImage.get('system:time_start'),
                                            'system:time_end': img.get('system:time_start')}));
    return change

"""# Define Study Site and Period"""

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

# Edit roi_name to include channels
roi_name_polygons = roi_name + "_S1_Channels_Landmask"
print(roi_name_polygons)

# ============================================== Start ============================================== #
# Study Site (lake boundary)
roi = ee.FeatureCollection(f"projects/floating-weed-manager/assets/Polygons/{roi_name_polygons}").geometry()

# Study Period
start, end = ee.Date(start_dt_str), ee.Date(end_dt_str)

# Filter by location
s1_col = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(roi) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

# Filter by date
s1_filtered_col = s1_col.filterDate(start, end.advance(1, 'days')) ## TODO .advance(1, 'days') ???

# Get size of image collection
size = s1_filtered_col.size().getInfo()
pprint('Number of images selected: ' + str(size))

if size >= 2:
    print('Two or more images available for change maps')
elif size == 1:
    print('Only one image available for change maps, fetching previous image as well...')
    # Get date of current image
    current_image = s1_filtered_col.first()
    current_image_date = ee.String(ee.Date(current_image.get('system:time_start')))#.format('YYYY-mm-DD')
    # Find the image immediately before the specified date
    previous_image = s1_col.filterDate('1980-01-01', current_image_date).sort('system:time_start', False).first()
    # Create an image collection containing the selected images
    s1_filtered_col = ee.ImageCollection([previous_image, current_image])
else:
    sys.exit()

# Pre-process imagery
s1_filtered_col = s1_filtered_col \
        .map(smoothImage) \
        .map(addWaterCorrected)

if s1_filtered_col.size().getInfo()==0: sys.exit()

# Mosaic tiles from same day
s1_filtered_col = mosaicByDate(s1_filtered_col)


# Visualise raw imagery
def visualise_sar(img):
    sarViz = {"opacity":1,"bands":["VV","VH","VV"],"min":-31.236695282711327,"max":-3.8713377563758455,"gamma":1}
    img_viz = img.visualize(**sarViz)
    return img_viz

raw_collection = s1_filtered_col.map(visualise_sar)

# Classify macrophytes
macrophyte_collection = s1_filtered_col.map(classify_macrophytes)

# Print out the number of images in the ImageCollection
count = macrophyte_collection.size().getInfo()
print("Number of radar macrophyte maps produced: ", count)

# Export the ImageCollection to Google Drive
image_list = macrophyte_collection.toList(count)
raw_image_list = raw_collection.toList(count)

dt_list = []
for i in range(0, count):
    image = ee.Image(image_list.get(i)).clip(roi).toFloat()
    raw_image = ee.Image(raw_image_list.get(i)).clip(roi).toFloat()
    this_time = str(ee.Date(ee.Image(image_list.get(i)).date()).format('YYYY-MM-dd').getInfo())
    print(f'{i}: {this_time}')
    dt_list.append(this_time)
    # export S1 macrophyte map
    task1 = ee.batch.Export.image.toDrive(
      image=image,
      description="_".join(['S1', roi_name, this_time]),
      region=roi,
      folder = 'classified_maps',
      scale=10,
      crs='EPSG:3857'
    )
    # export raw S1 image
    task2 = ee.batch.Export.image.toDrive(
      image=raw_image,
      description="_".join(['S1_Raw', roi_name, this_time]),
      region=roi,
      folder = 'raw_images', # new folder in "cron_output" folder
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

"""# Create Change Maps
> Change maps between subsequent images

## Apply Function
"""

# Convert ImageCollection to list
macrophyte_list = macrophyte_collection.toList(macrophyte_collection.size())

# For each image calculate change between subsequent image and current image
  # Change classes: (CHECK ALL THIS)
      # NA: clipped
      # 0: No data (partial image?)
      # 1: water -> land
      # 2: land -> water
      # 3: land -> land
      # 4: water -> water

change_list = macrophyte_list.map(create_change_collection)
# print(change_list)


# Remove first image (no change)
change_list = change_list.splice(0,1)

# print out the number of images in the ImageCollection
change_count = change_list.size().getInfo()
print("Number of radar change maps produced: ", change_count)

# export the ImageCollection to Google Drive
image_list = change_list

for i in range(0, change_count):
    image = ee.Image(image_list.get(i)).clip(roi).toFloat()
    name = "_".join([
      'S1_Change', 
      roi_name, 
      str(ee.Date(ee.Image(image_list.get(i)).get('system:time_start')).format('YYYY-MM-dd').getInfo()),
      str(ee.Date(ee.Image(image_list.get(i)).get('system:time_end')).format('YYYY-MM-dd').getInfo())
    ])
    task = ee.batch.Export.image.toDrive(
      image=image,
      description=name,
      region=roi,
      folder = 'change_maps',
      scale=10,
      crs='EPSG:3857'
    )
    task.start()
    print(task.status())

# Create Timeseries

# Function to calculate total change per change image in a single ROI
change_vals = [1, 2, 3, 4]
change_names = ['gainVeg', 'gainWater', 'stayVeg', 'stayWater']

def change_calc(img):
  # Remap to change types and get area (m2)
  change_count = img.eq(change_vals).rename(change_names)
  change_area = change_count.multiply(ee.Image.pixelArea()) #m2

  # Reduce to get total area in ROI
  total_area = change_area.reduceRegion(**{
    'reducer': ee.Reducer.sum(),
    'geometry': roi,
    'scale': 10,
    'maxPixels': 1e11,
    'bestEffort': True
  })
  # Add results to featurecollection including time property
  return ee.Feature(None, { 'gainVeg': ee.Number(total_area.get('gainVeg')),
                            'gainWater': ee.Number(total_area.get('gainWater')),
                            'stayVeg': ee.Number(total_area.get('stayVeg')),
                            'stayWater': ee.Number(total_area.get('stayWater')),
                            'system_time_start': img.get('system:time_start'),
                            'system_time_end': img.get('system:time_end')}) #create featurecollection

change_ftC = ee.FeatureCollection(ee.ImageCollection(change_list).map(change_calc))

# print(change_ftC.limit(3))

# spreadsheetId = '1p_p3i3bKPXM3UI_XoB1_YZjpmC4bAm13d58r3BxoILY' # S1_Change_Timeseries_LakeO_2022
spreadsheetId = '1eYOj0L4U1sOK1pwNWBTJjHbbSo3Hfpd_r1Drt-mB7C0' # S1_Change_Timeseries_LakeO_2023

def get_last_row_color(service, spreadsheetId):
    result = service.spreadsheets().get(spreadsheetId=spreadsheetId, ranges="Sheet1!A:A", includeGridData=True).execute()
    sheetId = result['sheets'][0]['properties']['sheetId']
    rowCount = result['sheets'][0]['properties']['gridProperties']['rowCount']
    sheet_data = result['sheets'][0]['data'][0]
    last_row_properties = sheet_data['rowData'][-1]['values'][0]['userEnteredFormat']
    bg = last_row_properties.get('backgroundColor', {}).copy()
    return sheetId, rowCount, bg

sheetId, rowCount, bg = get_last_row_color(sheet_service, spreadsheetId)
bg['blue'] = (bg['blue'] + 0.5) % 1

# Select columns
selectors = ['gainVeg','gainWater','stayVeg','stayWater','system_time_end','system_time_start']
rows = ee_to_pandas(change_ftC, col_names=selectors).values.tolist()

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

# # Export featurecollection
# name = "_".join([
#   'S1_Change_Timeseries',
#   roi_name,
#   str(start.format('YYYY-MM-dd').getInfo()),
#   str(end.format('YYYY-MM-dd').getInfo())
# ])
# task = ee.batch.Export.table.toDrive(
#   collection = change_ftC,
#   selectors = selectors,
#   description = name,
#   folder = 'timeseries',
#   fileFormat = 'CSV'
# )
# 
# task.start()
# print(task.status())
"""# S1 Channel blockage by waterbody code"""
# Quite computationally expensive, so the script might struggle on larger batches?

# Import the channel waterbody codes
waterbody_codes = ee.FeatureCollection("projects/floating-weed-manager/assets/Polygons/LakeO_Waterbody_codes_channels")

# Get long-term water landmask (channels only or there are some weird overlaps between this and the lake)
longterm_water = ee.Image('projects/floating-weed-manager/assets/Land_masks/LandMask_VH_' + roi_name + '_Waterbody_codes_channels_1yr')
    
def blockage_detection(img):
    # Define a function to process each feature in the ROI feature collection
    def processFeature(feature):
        # Get the geometry of the feature
        roi_code = ee.FeatureCollection([feature.geometry()])
        # Get landmask
        # Count water objects in long-term image
        roi_longterm_water = longterm_water.clip(roi_code).selfMask()  # test_roi
        # # Define a kernel for connectedComponents with 8-connectedness (diags too) (test)
        # kernel = ee.Kernel.fixed(3, 3, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # # Define a kernel for connectedComponents with 4-connectedness (plus) (test)
        # kernel = ee.Kernel.fixed(3, 3, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        # Create connected components (objects)
        longterm_objects = roi_longterm_water.connectedComponents(
            connectedness=ee.Kernel.plus(1),
            maxSize=1024
        )
        # Count water objects in longterm image
        longterm_objectCount = longterm_objects.select('labels').reduceRegion(
            reducer=ee.Reducer.countDistinct(),
            geometry= roi_code,
            scale= 10,  # Adjust the scale according to your data's resolution
            maxPixels= 1e9,  # You may need to adjust this value based on the image size
            tileScale= 4  # higher is lower memory but higher processing
        ).get('labels')
        # Get current water and mask to the current waterbody code and to the longterm water mask
        roi_current_water = img.Not().selfMask().updateMask(roi_longterm_water)
        # Create connected components (objects)
        current_objects = roi_current_water.connectedComponents(
            connectedness=ee.Kernel.plus(1),
            maxSize=1024
        )
        # Count water objects in current image
        current_objectCount = current_objects.select('labels').reduceRegion(
            reducer=ee.Reducer.countDistinct(),
            geometry=roi_code.geometry(),
            scale=10,
            maxPixels=1e9,
            tileScale=4
        ).get('labels')
        # Check if a blockage is detected
        blockage_detected = ee.Number(current_objectCount).gt(ee.Number(longterm_objectCount))
        # Count number of blockages
        num_blockages = ee.Number(current_objectCount).subtract(ee.Number(longterm_objectCount)) #.subtract(1) - why did I add this?
        # Save the S1 image date, blockage detection, and number of blockages
        feature = feature.set({
            'WaterbodyCode':feature.get('Name'),
            'ImageDate':img.date().format('YYYY-MM-dd'),
            'BlockageDetected':blockage_detected,
            # 'CurrentObjectCount':current_objectCount,
            # 'longtermObjectCount':longterm_objectCount,
            'NumberOfBlockages':num_blockages
        })
        return feature
    
    # Map the processFeature function over the ROI feature collection
    processed_roi = waterbody_codes.map(processFeature)
    return processed_roi

blockage_collection = ee.FeatureCollection(macrophyte_collection.map(blockage_detection)).flatten()

# Print the resulting feature collection
# print('Processed ROI:', processed_roi)

# Export featurecollection
selectors = ['WaterbodyCode', 'ImageDate', 'BlockageDetected','NumberOfBlockages']
#-------------------------------------------------------------------------------------------
# spreadsheetId = '1Fx_XS8h0Z-UOl2pyw8rwTCkYQA1X0jni07qoU_5d3sg' # S1_Channel_Blockage_2022
spreadsheetId = '1YubM2aw2gyAFAyOZqWORWWMoajSEVo--tNbRBR_CvhY' # S1_Channel_Blockage_2023
#-------------------------------------------------------------------------------------------
sheetId, rowCount, bg = get_last_row_color(sheet_service, spreadsheetId) # [201,218,248], [0.78823529, 0.85490196, 0.97254902]
bg['blue'] = (bg['blue'] + 0.5) % 1

rows = ee_to_pandas(blockage_collection, col_names=selectors).values.tolist()
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
print('append blockage timeserie rows')

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

# task = ee.batch.Export.table.toDrive(collection = blockage_collection,
#                                      selectors = selectors,
#                                      description = 'Blockage_roi_test',
#                                      folder = 'test_folder',
#                                      fileFormat = 'CSV')
# task.start()
# print(task.status())
