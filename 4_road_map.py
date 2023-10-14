#%%
import geopandas as gpd
import os
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from pandas.testing import assert_frame_equal
import seaborn as sns
import argparse
import warnings
from shapely.geometry import LineString, Point, Polygon

from geopy.distance import geodesic 
import math
import geopandas
import warnings
warnings.filterwarnings("ignore")




locs = ['Hand', 'Bag', 'Hips', 'Torso']

  # 'Hand'
dataset = 'valid'

label_map = {
    1: 'Still',
    2: 'Walking',
    3: 'Run',
    4: 'Bike',
    5: 'Car',
    6: 'Bus',
    7: 'Train',
    8: 'Subway'
}

filenames = {
    'train': {
        'Hand': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Location_new.pkl',
            # 进行标签匹配后的数据，时间戳是1HZ的label数据
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS_new.pkl',
        },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS_new.pkl',
        },
        'Hips': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS_new.pkl',
        },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/GPS_new.pkl',
        },
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Label.pkl'
    },
    'valid': {
        'Hand': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS_new.pkl',
        },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/GPS_new.pkl',
        },

        'Hips': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS_new.pkl',
        },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/GPS_new.pkl',
        },
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Label.pkl',
    },
    'test': {
        'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Location.pkl',
        'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Location_new.pkl',
        'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Mag.pkl',
        'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Gyr.pkl',
        'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Acc.pkl',
        'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/GPS.pkl',
        'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/GPS_new.pkl',
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Label.pkl'
    }
}

dics = {'raliways_class': {'rail': 1,
  'narrow_gauge': 2,
  'subway': 3,
  'light_rail': 4,
  'tram': 5,
  'miniature_railway': 6,
  'monorail': 7,
  'nan': 0},
 'transport_class': {'bus_stop': 1,
  'bus_station': 2,
  'taxi': 3,
  'railway_station': 4,
  'airport': 5,
  'tram_stop': 6,
  'airfield': 7,
  'apron': 8,
  'ferry_terminal': 9,
  'helipad': 10,
  'nan': 0},
 'traffic_class': {'crossing': 1,
  'parking': 2,
  'street_lamp': 3,
  'parking_bicycle': 4,
  'traffic_signals': 5,
  'parking_multistorey': 6,
  'fuel': 7,
  'mini_roundabout': 8,
  'speed_camera': 9,
  'turning_circle': 10,
  'lock_gate': 11,
  'marina': 12,
  'service': 13,
  'parking_underground': 14,
  'pier': 15,
  'stop': 16,
  'motorway_junction': 17,
  'weir': 18,
  'slipway': 19,
  'nan': 0},
 'landuse_class': {'residential': 1,
  'park': 2,
  'retail': 3,
  'grass': 4,
  'commercial': 5,
  'cemetery': 6,
  'forest': 7,
  'scrub': 8,
  'industrial': 9,
  'nature_reserve': 10,
  'farmland': 11,
  'meadow': 12,
  'recreation_ground': 13,
  'allotments': 14,
  'orchard': 15,
  'farmyard': 16,
  'quarry': 17,
  'heath': 18,
  'military': 19,
  'vineyard': 20,
  'nan': 0},
 'roads_class': {'path': 1,
  'residential': 2,
  'unclassified': 3,
  'cycleway': 4,
  'footway': 5,
  'steps': 6,
  'tertiary': 7,
  'primary': 8,
  'pedestrian': 9,
  'trunk': 10,
  'service': 11,
  'secondary': 12,
  'primary_link': 13,
  'tertiary_link': 14,
  'bridleway': 15,
  'track': 16,
  'trunk_link': 17,
  'track_grade2': 18,
  'track_grade1': 19,
  'motorway': 20,
  'living_street': 21,
  'track_grade4': 22,
  'track_grade3': 23,
  'track_grade5': 24,
  'secondary_link': 25,
  'motorway_link': 26,
  'nan': 0},
 'roads_code': {'515.0': 1,
  '512.0': 2,
  '511.0': 3,
  '514.0': 4,
  '513.0': 5,
  'nan': 0}}


#使用了四个数据集:路网、铁路、transport、traffic、landuse
dirs = {
    'raliways': ['/DATA2/lvxiaoling/limengyuan/SHL2023/lmy/england-latest-free.shp/gis_osm_railways_free_1.pkl',0.0008],
    'transport': ['/DATA2/lvxiaoling/limengyuan/SHL2023/lmy/england-latest-free.shp/gis_osm_transport_free_1.pkl',0.0003],
    'traffic': ['/DATA2/lvxiaoling/limengyuan/SHL2023/lmy/england-latest-free.shp/gis_osm_traffic_free_1.pkl',0.0003],
    'landuse': ['/DATA2/lvxiaoling/limengyuan/SHL2023/lmy/england-latest-free.shp/gis_osm_landuse_a_free_1.pkl',0.0002],
    'roads':['/DATA2/lvxiaoling/limengyuan/SHL2023/lmy/england-latest-free.shp/gis_osm_roads_free_1.pkl',0.0003],

}
for dir in dirs.keys():
    print(dir,'loading')
    dirs[dir].append(pd.read_pickle(dirs[dir][0]))

def get_osm(dataset='valid',loc='Hand'):
    
    if dataset=='test':
        data_dir = filenames[dataset]['Location_new']
    else: 
        data_dir = filenames[dataset][loc]['Location_new']
    data = pd.read_pickle(data_dir)
    data_gdp =  geopandas.GeoDataFrame(
        data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))
    print(data_dir)
    for dir in dirs.keys():
        print(dir)
        osm_data = dirs[dir][2]


        data_gdp1 = data_gdp.sjoin_nearest(osm_data, how="left", max_distance =dirs[dir][1], distance_col="distances")
        print('join finish...')
        data_gdp1 = data_gdp1.sort_values(by=['idx', 'distances'])
        data_gdp1 = data_gdp1.drop_duplicates(subset='idx', keep='first')
        data['{}_class'.format(dir)] = data_gdp1['fclass'].astype(str).values

        if [dir][0] in ['roads']:

            data['{}_code'.format(dir)] = (data_gdp1['code']//10).astype(str)
    for class_i in [ 'raliways_class','transport_class','traffic_class','landuse_class','roads_class','roads_code']:
        data[class_i] = data[class_i].map(dics[class_i])

    data.to_pickle(data_dir)
for dataset in ['valid','train']:
    for loc in [ 'Hand']:#['Hand', 'Bag', 'Hips', 'Torso']:
        print(dataset, loc)
        get_osm(dataset=dataset, loc=loc)
dataset = 'test'
get_osm(dataset=dataset, loc=loc)

# %%
