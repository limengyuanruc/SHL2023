#%%
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
import progressbar
warnings.filterwarnings('ignore')
#nohup python -u 2_GPS_location_data.py > 2_GPS_location_data.out  2>&1 &
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

'''
data_gps = pd.read_pickle(filenames['train']['Bag']['Location'])
a = pd.DataFrame((data_gps['timestamp']//1000).diff().value_counts().head(10))
a['ratio'] = a['timestamp']/a['timestamp'].sum()
a
'''
#data_train_label = pd.read_pickle(filenames['train']['Label'])
#%%
#首先对轨迹进行分段，从下面的图看出，99%+的数据是间隔0、1、2、3的，因此我们把间隔>3秒视为GPS数据缺失
# nohup python -u process.py > nohup.out 2>&1 &


#轨迹分段：
def pro_gps(data_dir,type='train'):
    print('startload...')
    print(data_dir)
    data = pd.read_pickle(data_dir)[['timestamp', 'accuracy', 'latitude', 'longitude', 'altitude', 'time']]#原始的数据
    
        
    print('end_load...')

    

    def get_trip(data):

        dif = (data['timestamp']//1000).diff()#差值
        start = 0
        new_trip=0
        data_new = []
        for i in range(1,len(dif)):
            if dif[i] > 3:
                #print(dif[i])
                ttt = data[start:i].reset_index(drop=True)
                ttt['gps_trip_id']=new_trip
                #print(ttt,len(ttt),len(ttt['time']))
                ttt['g'] = (ttt['time'].diff() > pd.Timedelta(milliseconds=100)).cumsum()

                ttt = ttt.groupby(['g']).last().reset_index()
                data_new.append(ttt)
                start = i
                new_trip+=1
        ttt = data[start:i]#因为现在的i就是最后一个了
        ttt['g'] = (ttt['time'].diff() > pd.Timedelta(milliseconds=100)).cumsum()
        ttt = ttt.groupby(['g']).last().reset_index()
        ttt['gps_trip_id']=new_trip
        data_new.append(ttt)
        #new_trip+=1
        data_new = pd.concat(data_new).reset_index()
        data_new = data_new.drop(['g','index'],axis=1)
        return data_new



    def compute_distance(fLat, fLon, sLat, sLon):#计算距离
        return geodesic((fLat,fLon), (sLat,sLon)).meters


    def compute_bearing(fLat, fLon, sLat, sLon):#计算方向
        y = math.sin(math.radians(sLon) - math.radians(fLon)) * math.cos(math.radians(sLat))
        x = math.cos(math.radians(fLat)) * math.sin(math.radians(sLat)) - \
            math.sin(math.radians(fLat)) * math.cos(math.radians(sLat)) \
            * math.cos(math.radians(sLon) - math.radians(fLon))
        bear =  (math.atan2(y, x) * 180. / math.pi + 360) % 360
        return bear * math.pi /180 #正北方向

    #速度和方向
    def spe_bear(x):
        #latitude1,longitude1,latitude,longitude,timestamp,timestamp1
        dt = (x['timestamp']-x['timestamp1'])/1000
        #print(x)
        spe = compute_distance(x['latitude1'],x['longitude1'],x['latitude'],x['longitude']) / dt
        bearing = compute_bearing(x['latitude1'],x['longitude1'],x['latitude'],x['longitude']) #朝向
        return spe,bearing


    #纵向加速度、横向加速度




    data_new = get_trip(data)
    data_new.loc[:,['speed','bearing']] = 0
    def get_spe_bear(data_temp):
        
        data0 = pd.DataFrame({'timestamp':[0],
                            'longitude':[0],
                            'latitude':[0]})
        data1 = data_temp[:-1][['timestamp','longitude','latitude']]
        data_1 = pd.concat([data0,data1]).reset_index(drop=True)#第一行往后移动
        data_1.columns = ['timestamp1','longitude1','latitude1']
        data2 = pd.concat([data_temp,data_1],axis=1)

        data3 = []
        bar = progressbar.ProgressBar()
        for idx,data_i in  bar(data2.groupby(['gps_trip_id'])):

            
            temp = data_i.reset_index(drop=True)
            temp[['speed','bearing']] = np.nan
            if len(temp)>1:
                #temp[['speed','bearing']] = 0
                temp.loc[1:,['speed','bearing']] = temp[1:].apply(spe_bear, axis=1, result_type="expand").values
                #print(temp)
            data3.append(temp)

        data_all = pd.concat(data3).reset_index(drop=True)
        data_all['speed'] = data_all['speed'].clip(upper=100)

        return data_all[['speed','bearing']]

    data_new.loc[:,['speed','bearing']] = get_spe_bear(data_new[['timestamp','longitude','latitude','gps_trip_id']])

    print('over....')





    def feat2(x):

        #latitude1,,'gps_trip_id','speed','bearing'
        dt = (x['timestamp']-x['timestamp1'])/1000
        #print(x)
        bearing_rate = (x['bearing']-x['bearing1']) / dt
        acc_lng = (x['speed']-x['speed1']) / dt #朝向
        acc_lat = x['speed'] * bearing_rate

        return acc_lng,acc_lat,bearing_rate

    def get_feat2(data_temp):

        data0 = pd.DataFrame({'timestamp':[0],
                            'speed':[0],
                            'bearing':[0]})
        data1 = data_temp[:-1][['timestamp','speed','bearing']]#第一行忽略

        data_1 = pd.concat([data0,data1]).reset_index(drop=True)
        data_1.columns = ['timestamp1','speed1','bearing1']

        data2 = pd.concat([data_temp,data_1],axis=1)

        data3 = []
        bar = progressbar.ProgressBar()
        for idx,data_i in  bar(data2.groupby(['gps_trip_id'])):
            #print(idx)
            temp = data_i.reset_index(drop=True)
            temp[['acc_lng','acc_lat','bearing_rate']] = np.nan
            if len(temp) >2:
            #temp[['acc_lng','acc_lat','bearing_rate']] = 0
                temp.loc[2:,['acc_lng','acc_lat','bearing_rate']] = temp[2:].apply(feat2, axis=1, result_type="expand").values
            #print(temp)
            data3.append(temp)

        data_all = pd.concat(data3).reset_index(drop=True)
        data_all['acc_lng'] = data_all['acc_lng'].clip(upper=4,lower=-4)
        data_all['acc_lat'] = data_all['acc_lat'].clip(upper=8,lower=-8)
        data_all['bearing_rate'] = data_all['bearing_rate'].clip(upper=6,lower=-6)
        return data_all[['acc_lng','acc_lat','bearing_rate']]


    data_new.loc[:,['acc_lng','acc_lat','bearing_rate']] = get_feat2(data_new[['timestamp','gps_trip_id','speed','bearing']])




    def get_feat3(data_temp):

        data0 = pd.DataFrame({'timestamp':[0],
                            'acc_lng':[0]})
        data1 = data_temp[:-1][['timestamp','acc_lng']]#

        data_1 = pd.concat([data0,data1]).reset_index(drop=True)
        data_1.columns = ['timestamp1','acc_lng1']

        data2 = pd.concat([data_temp,data_1],axis=1)

        data3 = []
        bar = progressbar.ProgressBar()
        for idx,data_i in  bar(data2.groupby(['gps_trip_id'])):
            #print(idx)
            temp = data_i.reset_index(drop=True)
            temp[['jert']] = np.nan
            if len(temp)>3:
                temp.loc[3:,['jert']] = ((temp['acc_lng']-temp['acc_lng1'])/(temp['timestamp']-temp['timestamp1']))[3:]
            #print(temp)
            data3.append(temp)

        data_all = pd.concat(data3).reset_index(drop=True)
        data_all['jert'] = data_all['jert'].clip(upper=0.07,lower=-0.07)
        return data_all[['jert']]
    data_new.loc[:,['jert']] = get_feat3(data_new[['timestamp','gps_trip_id','acc_lng']])
    return data_new



locs = ['Hand', 'Bag', 'Hips', 'Torso']

#data_dir = filenames['valid']['Hand']['Location']


for type in ['train','valid']:
    times = pd.DataFrame()
    for loc in locs:
        filename = filenames[type][loc]['Location']
        
        data = pro_gps(filename)
        data.to_pickle(filename)#保存

filename = filenames['test']['Location']
data = pro_gps(filename)
data.to_pickle(filename)#保存


'''


data_mew2 = []
bar = progressbar.ProgressBar() 
for temp1 in bar(data_new.groupby(['gps_trip_id'])):
    temp = temp1[1].reset_index(drop=True)
    time_predict_iter = pd.DataFrame()
    time_predict_iter['timestamp1'] = pd.date_range(start = temp['timestamp1'].min(), end = temp['timestamp1'].max(), freq = '1 s')#生成逐秒的时间序列
    temp_new =  pd.merge(time_predict_iter,temp, on='timestamp1', how='left')#merge
    temp_new = temp_new.groupby(['timestamp1']).mean().reset_index()
    #对重复数据处理
    temp_new[['uid', 'upload_type', 'gps_trip_id','tid', 'new_trip','city']] = temp[['uid', 'upload_type', 'gps_trip_id','tid', 'new_trip','city']].values[0]
    temp_new[['latitude', 'longitude', 'roll', 'pitch', 'yaw', 'speed', 'speed_gnss', 'speed_hypot', 
              'accel_lateral', 'speed_hypot_sub_gnss', 'accel_lateral_gnss', 'accel_lateral_hypot', 
              'accel_longitudinal', 'accel_longitudinal_gnss', 'accel_longitudinal_hypot', 'alpha', 
              'omega', 'orientation_type', 'illuminance', 'imu_accel_l2', 'bearing', 'bearing_gnss', 
              'bearing_hypot']] = temp_new[['latitude', 'longitude', 'roll', 'pitch', 'yaw', 'speed', 'speed_gnss', 'speed_hypot', 
              'accel_lateral', 'speed_hypot_sub_gnss', 'accel_lateral_gnss', 'accel_lateral_hypot', 
              'accel_longitudinal', 'accel_longitudinal_gnss', 'accel_longitudinal_hypot', 'alpha', 
              'omega', 'orientation_type', 'illuminance', 'imu_accel_l2', 'bearing', 'bearing_gnss', 
              'bearing_hypot']].interpolate()
    data_mew2.append(temp_new)
data_mew2 = pd.concat(data_mew2)


'''
