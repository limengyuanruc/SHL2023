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
def fit_transform(data):
    result_dict = {}
    data_u = data.unique()
    for idx, item in enumerate(data_u):
        if item!= 'nan':
            result_dict[item] = idx+1

    result_dict['nan'] = 0  # 添加对应字符串 '0' 的索引映射
    return data.map(result_dict)



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
        data['{}_class'.format(dir)] = fit_transform(data_gdp1['fclass'].astype(str)).values

        if [dir][0] in ['roads']:

            data['{}_code'.format(dir)] =fit_transform( (data_gdp1['code']//10).astype(str))
    
    data.to_pickle(data_dir)
for dataset in ['valid','train']:
    for loc in ['Hand', 'Bag', 'Hips', 'Torso']:#['Hand', 'Bag', 'Hips', 'Torso']:
        print(dataset, loc)
        get_osm(dataset=dataset, loc=loc)
dataset = 'test'
get_osm(dataset=dataset, loc=loc)

# %%
