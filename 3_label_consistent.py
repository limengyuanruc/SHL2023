
# Code that might generate warnings
#%%
import warnings
# nohup python -u 3_label_consistent.py > 3_label_consistent.out  2>&1 &
warnings.filterwarnings("ignore")  # Ignore all warnings
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
from geopy.distance import geodesic 
import math
import matplotlib.colors as mcolors
import _pickle as cPickle

#
loc = ['Hand', 'Bag', 'Hips', 'Torso']
filenames = {
    'train': {
        'Hand': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS.pkl',
            },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS.pkl',
            },
        'Hips': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS.pkl',
            },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/GPS.pkl',
            },
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Label.pkl'
        },
    'valid': {
        'Hand': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS.pkl',
            },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/GPS.pkl',
            },

        'Hips': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS.pkl',
            },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Acc.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/GPS.pkl',
           },
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Label.pkl',
        },
    'test': {
        'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Location.pkl',
        'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Mag.pkl',
        'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Gyr.pkl',
        'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Acc.pkl',
        'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/GPS.pkl',
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Label.pkl'
        }
}


# %%

index_start = 0
print('train GPS....')
train_label = pd.read_pickle(filenames['train']['Label'])
train_label_1HZ = train_label[['timestamp','time','label','trajectory_id','idx','label_idx']]
train_label_1HZ['timestamp'] = train_label_1HZ['timestamp']//1000 * 1000 
train_label_1HZ = train_label_1HZ.drop_duplicates(subset='timestamp', keep='first',)

for i in range(4):
    filename = filenames['train'][loc[i]]['GPS']
    print(filename)
    gps =  pd.read_pickle(filename)
    gps = gps.rename(columns={'timestamp': 'timestamp1'})
    gps['time1'] = gps['time']
    gps['index1'] = np.arange(len(gps))
    gapl = pd.merge_asof(train_label_1HZ,gps,on='time',tolerance=pd.Timedelta('1s'))
    print(loc[i],'GPS占比',1- sum(gapl['num'].isna())/len(gapl))
    gapl.to_pickle(filename.replace('GPS','GPS_new'))
    '''
    ['timestamp', 'time', 'label', 'trajectory_id', 'idx', 'timestamp1',
       'accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert', 'time1',
       'index1']
    '''
    filename = filenames['train'][loc[i]]['Location']
    print(filename)
    gps =  pd.read_pickle(filename)
    gps = gps.rename(columns={'timestamp': 'timestamp1'})
    gps['time1'] = gps['time']
    gps['index1'] = np.arange(len(gps))

    gapl = pd.merge_asof(train_label_1HZ,gps,on='time',tolerance=pd.Timedelta('1s'))
    # 按照GPS_id进行经纬度等信息的线性补全

    gapl[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']] = gapl.groupby('trajectory_id')[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']].apply(lambda x: x.interpolate())
    print(loc[i],'Location占比',1- sum(gapl['accuracy'].isna())/len(gapl))
    gapl.to_pickle(filename.replace('Location','Location_new'))




print('valid GPS....')
valid_label = pd.read_pickle(filenames['valid']['Label'])
train_label_1HZ = valid_label[['timestamp','time','label','trajectory_id','idx','label_idx']]
train_label_1HZ['timestamp'] = train_label_1HZ['timestamp']//1000 * 1000 
#train_label_1HZ = train_label_1HZ.drop_duplicates(subset='timestamp', keep='first',)
train_label_1HZ['day'] = train_label_1HZ['timestamp']//(1000 * 60 *60 * 24)
day1=np.where(train_label_1HZ['day']==17350)[0][0]#第二段的开始
day2=np.where(train_label_1HZ['day']==17354)[0][-1]+1#第三段的开始
train_label_1HZ_1 = train_label_1HZ[:day1]
train_label_1HZ_2 = train_label_1HZ[day1:day2]
train_label_1HZ_3 = train_label_1HZ[day2:]
train_label_1HZ_1 = train_label_1HZ_1.drop_duplicates(subset='timestamp', keep='first',)
train_label_1HZ_2 = train_label_1HZ_2.drop_duplicates(subset='timestamp', keep='first',)
train_label_1HZ_3 = train_label_1HZ_3.drop_duplicates(subset='timestamp', keep='first',)

for i in range(4):

    filename = filenames['valid'][loc[i]]['GPS']
    print(filename)
    valid_location =  pd.read_pickle(filename)
    valid_location = valid_location.rename(columns={'timestamp': 'timestamp1'})
    valid_location['time1'] = valid_location['time']
    valid_location['index1'] = np.arange(len(valid_location))
    valid_location['day'] =  valid_location['timestamp1']//(1000 * 60 *60 * 24)
    day1=np.where(valid_location['day']==17350)[0][0]#第二段的开始
    day2=np.where(valid_location['day']==17354)[0][-1]+1#第三段的开始
    valid_location_1 = pd.merge_asof(train_label_1HZ_1,valid_location[:day1],on='time',tolerance=pd.Timedelta('1s'))
    valid_location_2 = pd.merge_asof(train_label_1HZ_2,valid_location[day1:day2],on='time',tolerance=pd.Timedelta('1s'))
    valid_location_3 = pd.merge_asof(train_label_1HZ_3,valid_location[day2:],on='time',tolerance=pd.Timedelta('1s'))
    valid_location= pd.concat([valid_location_1,valid_location_2,valid_location_3])
    valid_location = valid_location.drop(['day_x','day_y'],axis=1)
    print(loc[i],'GPS占比',1- sum(valid_location['num'].isna())/len(valid_location))
    valid_location.to_pickle(filename.replace('GPS','GPS_new'))


    filename = filenames['valid'][loc[i]]['Location']
    print(filename)
    valid_location =  pd.read_pickle(filename)
    valid_location = valid_location.rename(columns={'timestamp': 'timestamp1'})
    valid_location['time1'] = valid_location['time']
    valid_location['index1'] = np.arange(len(valid_location))
    valid_location['day'] =  valid_location['timestamp1']//(1000 * 60 *60 * 24)
    day1=np.where(valid_location['day']==17350)[0][0]#第二段的开始
    day2=np.where(valid_location['day']==17354)[0][-1]+1#第三段的开始
    valid_location_1 = pd.merge_asof(train_label_1HZ_1,valid_location[:day1],on='time',tolerance=pd.Timedelta('1s'))
    valid_location_2 = pd.merge_asof(train_label_1HZ_2,valid_location[day1:day2],on='time',tolerance=pd.Timedelta('1s'))
    valid_location_3 = pd.merge_asof(train_label_1HZ_3,valid_location[day2:],on='time',tolerance=pd.Timedelta('1s'))
    valid_location= pd.concat([valid_location_1,valid_location_2,valid_location_3])
    valid_location = valid_location.drop(['day_x','day_y'],axis=1)


    # 按照GPS_id进行经纬度等信息的线性补全
    valid_location[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']] = valid_location.groupby('trajectory_id')[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']].apply(lambda x: x.interpolate())
    print(loc[i],'Location占比',1- sum(valid_location['accuracy'].isna())/len(valid_location))
    valid_location.to_pickle(filename.replace('Location','Location_new'))


#%%

print('test GPS....')
train_label = pd.read_pickle(filenames['test']['Label'])
train_label_1HZ = train_label[['timestamp','time','label','trajectory_id','idx','label_idx']]
train_label_1HZ['timestamp'] = train_label_1HZ['timestamp']//1000 * 1000 
train_label_1HZ = train_label_1HZ.drop_duplicates(subset='timestamp', keep='first',)

filename = filenames['test']['GPS']
print(filename)
gps =  pd.read_pickle(filename)
gps = gps.rename(columns={'timestamp': 'timestamp1'})
gps['time1'] = gps['time']
gps['index1'] = np.arange(len(gps))
gapl = pd.merge_asof(train_label_1HZ,gps,on='time',tolerance=pd.Timedelta('1s'))
print('GPS占比',1- sum(gapl['num'].isna())/len(gapl))
gapl.to_pickle(filename.replace('GPS','GPS_new'))

filename = filenames['test']['Location']
print(filename)
gps =  pd.read_pickle(filename)
gps = gps.rename(columns={'timestamp': 'timestamp1'})
gps['time1'] = gps['time']
gps['index1'] = np.arange(len(gps))
gapl = pd.merge_asof(train_label_1HZ,gps,on='time',tolerance=pd.Timedelta('1s'))

# 按照GPS_id进行经纬度等信息的线性补全
gapl[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
    'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']] = gapl.groupby('trajectory_id')[['accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
    'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']].apply(lambda x: x.interpolate())
print('Location占比',1- sum(gapl['accuracy'].isna())/len(gapl))
gapl.to_pickle(filename.replace('Location','Location_new'))
# %%
