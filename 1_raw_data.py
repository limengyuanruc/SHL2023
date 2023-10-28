'''
label文件：在train和valid数据集中，四个位置的label文件是一样的，因此只需采用一个。数据保存为pandas数据框，列名为['timestamp','label'，'time' , 'weekday', 'hour', 'minute', 'second', 'allminute'] （time表示英国时间，allminute：比如2点表示2*60=120）。测试集的时间戳被处理了，显示2049年.....​
2.
手机传感器文件（加速度、陀螺仪、磁力计）：时间戳与label时间戳是一致的。数据保存为pandas数据框，列名为['timestamp','acc_x','acc_y','acc_z']​
3.
GPS Location数据: 删除2,3列，重命名为 ['timestamp','accuracy','latitude','longitude','altitude','time']​
4.
GPS reception数据:  删除2,3列，重命名 ['timestamp','num','satellite','snr','azimuth','elevation','time'],其中'satellite','snr','azimuth','elevation'为列表

'''
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
parser = argparse.ArgumentParser()
parser.add_argument('--gps_reception_process', type=bool, default=False) 
parser.add_argument('--gps_process', type=bool, default=False) 
parser.add_argument('--sensor_process', type=bool, default=False) 
parser.add_argument('--label_process', type=bool, default=False) 

args = parser.parse_args([])


locs = ['Hand', 'Bag', 'Hips', 'Torso']
motion = ['Acc','Gyr','Mag',]#加速度计、陀螺仪、磁力计
radio = ['Location', 'GPS']

filenames = {'train': {'Hand': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Label.txt'},
  'Bag': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Label.txt'},
  'Hips': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Label.txt'},
  'Torso': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Label.txt'}},
 'valid': {'Hand': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/Label.txt'},
  'Bag': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Label.txt'},
  'Hips': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/Label.txt'},
  'Torso': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location.txt',
   'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Mag.txt',
   'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Gyr.txt',
   'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Acc.txt',
   'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/GPS.txt',
   'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Label.txt'}},
 'test': {'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Location.txt',
  'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Mag.txt',
  'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Gyr.txt',
  'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Acc.txt',
  'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/GPS.txt',
  'Label_idx': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Label_idx.txt'}}
#%%
def timestamp_to_strtime(timestamp):
    timezone = pytz.timezone('Europe/London')#采用英国的时区
    local_str_time = datetime.fromtimestamp(timestamp / 1000.0,tz=timezone).strftime('%Y-%m-%d %H:%M:%S.%f') 
    #local_dt_time = datetime.fromtimestamp(timestamp / 1000.0)
    return local_str_time

def time_fun(data,all =True):
    data['time'] =  pd.to_datetime(data['timestamp'].apply(timestamp_to_strtime))
    if all:
        data['weekday'] = data['time'].dt.dayofweek
        data['hour'] = data['time'].dt.hour
        data['minute'] = data['time'].dt.minute
        data['second'] = data['time'].dt.second
        data['allminute'] = data['hour'] * 60 + data['minute']
    return data

def read_acc_sensors(file_name):
    print('loading',file_name)
    data = pd.read_csv(file_name,sep='\t',header=None)
    data.columns = ['timestamp','acc_x','acc_y','acc_z']
    return data 

def read_gyr_sensors(file_name):
    print('loading',file_name)
    data = pd.read_csv(file_name,sep='\t',header=None)
    data.columns = ['timestamp','gyr_x','gyr_y','gyr_z']
    return data 

def read_mag_sensors(file_name):
    print('loading',file_name)
    data = pd.read_csv(file_name,sep='\t',header=None)
    data.columns = ['timestamp','mag_x','mag_y','mag_z']
    return data
    

def read_label(file_name):
    print('loading',file_name)
    data_label = pd.read_csv(file_name,sep='\t',header=None)
    if data_label.shape[1]==1:#测试集
        data_label.columns = ['timestamp']
    else:
        data_label.columns = ['timestamp','label']
    data_label = time_fun(data_label)
    return data_label

def read_gps(filename):
    print('loading',filename)
    data = pd.read_csv(filename,sep=' ',header=None)#时间戳大约1s收集的
    data = data[[0,3,4,5,6]]
    data.columns = ['timestamp','accuracy','latitude','longitude','altitude']

    data = time_fun(data,all =False)
    return data


def read_reception_gps(filename):
    print('loading',filename)

    file = open(filename, "r")  # "r" 表示以只读模式打开文件
    # 读取文件内容
    content = file.readlines()
    # 关闭文件
    file.close()
    content = [i.replace('\n','').split(' ') for i in content]

    def fun(lls):
        temp_list = [int(lls[0]),int(lls[-1])]
        o_list = lls[3:-1]
        satellite = [int(j) for i,j in enumerate(o_list) if i % 4 ==0]
        Snr= [float(j) for i,j in enumerate(o_list) if i % 4 ==1]
        Azimuth= [float(j) for i,j in enumerate(o_list) if i % 4 ==2]
        elevation= [float(j) for i,j in enumerate(o_list) if i % 4 ==3]
        return temp_list+[satellite,Snr,Azimuth,elevation]
    content_lis =  [fun(i) for i in content]
    data = pd.DataFrame(content_lis)
    data.columns = ['timestamp','num','satellite','snr','azimuth','elevation']
    data = time_fun(data,all =False)
    return data





#%%
if args.gps_reception_process:
    for type in ['train','valid']:
        times = pd.DataFrame()
        for loc in locs:
            filename = filenames[type][loc]['GPS']
            data = read_reception_gps(filename)
            data.to_pickle(filename.replace('.txt','.pkl'))#保存
            times[loc] = data['timestamp']

        if times[[locs[0]]].equals(times[[locs[1]]]) and times[[locs[0]]].equals(times[[locs[2]]]) and times[[locs[0]]].equals(times[[locs[3]]]):
            print("Raw gps_reception time are the same")
        else:
            print("Raw gps_reception time are not the same")
        if (times[[locs[0]]]//1000).equals(times[[locs[1]]]//1000) and (times[[locs[0]]]//1000).equals((times[[locs[2]]]//1000)) and (times[[locs[0]]]//1000).equals((times[[locs[3]]]//1000)):
            print("Raw gps_reception time//1000 are the same")
        else:
            print("Raw gps_reception time//1000 are not the same")
    filename = filenames['test']['GPS']
    data = read_reception_gps(filename)
    data.to_pickle(filename.replace('.txt','.pkl'))#保存

#%%

if args.gps_process:
    for type in ['train','valid']:
        times = pd.DataFrame()
        for loc in locs:
            filename = filenames[type][loc]['Location']
            data = read_gps(filename)
            data.to_pickle(filename.replace('.txt','.pkl'))#保存
            times[loc] = data['timestamp']

        if times[[locs[0]]].equals(times[[locs[1]]]) and times[[locs[0]]].equals(times[[locs[2]]]) and times[[locs[0]]].equals(times[[locs[3]]]):
            print("Raw gps time are the same")
        else:
            print("Raw gps time are not the same")
        if (times[[locs[0]]]//1000).equals(times[[locs[1]]]//1000) and (times[[locs[0]]]//1000).equals((times[[locs[2]]]//1000)) and (times[[locs[0]]]//1000).equals((times[[locs[3]]]//1000)):
            print("Raw gps time//1000 are the same")
        else:
            print("Raw gps time//1000 are not the same")
    filename = filenames['test']['Location']
    data = read_gps(filename)
    data.to_pickle(filename.replace('.txt','.pkl'))#保存




#%%
# 2\
if args.sensor_process:
    for type in ['train','valid']:
        Hand = filenames[type]['Hand']['Label']#['Hand', 'Bag', 'Hips', 'Torso']
        #/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/Label.txt
        df_Hand = read_label(Hand)#不同位置标签文件相同
        df_Hand.to_pickle(Hand.replace('Hand/Label.txt','Label.pkl'))#保存
        
        #接下来对传感器Acc文件进行操作
        for loc in locs:#['Hand', 'Bag', 'Hips', 'Torso']
            filename = filenames[type][loc]['Acc']
            data = read_acc_sensors(filename)
            if data[['timestamp']].equals(df_Hand[['timestamp']]):
                print(type,loc,'Acc timestamp match')
            else:
                print(type,loc,'Acc NOT timestamp match!!!!')
            data.to_pickle(filename.replace('.txt','.pkl'))

        #接下来对传感器Gyr文件进行操作
        for loc in locs:#['Hand', 'Bag', 'Hips', 'Torso']
            filename = filenames[type][loc]['Gyr']
            data = read_gyr_sensors(filename)
            if data[['timestamp']].equals(df_Hand[['timestamp']]):
                print(type,loc,'Gyr timestamp match')
            else:
                print(type,loc,'Gyr NOT timestamp match!!!!')
            data.to_pickle(filename.replace('.txt','.pkl'))

        #接下来对传感器Mag文件进行操作
        for loc in locs:#['Hand', 'Bag', 'Hips', 'Torso']
            filename = filenames[type][loc]['Mag']
            data = read_mag_sensors(filename)
            if data[['timestamp']].equals(df_Hand[['timestamp']]):
                print(type,loc,'Mag timestamp match')
            else:
                print(type,loc,'Mag NOT timestamp match!!!!')
            data.to_pickle(filename.replace('.txt','.pkl'))
        #测试集
    type = 'test'
    Hand = filenames[type]['Label_idx']
    df_Hand = read_label(Hand)
    df_Hand.to_pickle(Hand.replace('Label_idx.txt','Label.pkl'))

    #接下来对传感器Acc文件进行操作
    filename = filenames[type]['Acc']
    data = read_acc_sensors(filename)
    if data[['timestamp']].equals(df_Hand[['timestamp']]):
        print(type,'Acc timestamp match')
    else:
        print(type,loc,'ACC NOT timestamp match!!!!')
    data.to_pickle(filename.replace('.txt','.pkl'))

    #接下来对传感器Gyr文件进行操作
    filename = filenames[type]['Gyr']
    data = read_gyr_sensors(filename)
    if data[['timestamp']].equals(df_Hand[['timestamp']]):
        print(type,'Gyr timestamp match')
    else:
        print(type,loc,'Gyr NOT timestamp match!!!!')
    data.to_pickle(filename.replace('.txt','.pkl'))

    #接下来对传感器Mag文件进行操作
    filename = filenames[type]['Mag']
    data = read_mag_sensors(filename)
    if data[['timestamp']].equals(df_Hand[['timestamp']]):
        print(type,'Mag timestamp match')
    else:
        print(type,loc,'Mag NOT timestamp match!!!!')
    data.to_pickle(filename.replace('.txt','.pkl'))





#1、判断标签文件是否一致。已完成 不同位置的标签是一样的。
if args.label_process:
    for type in ['train','valid']:
        Hand = filenames[type]['Hand']['Label']#['Hand', 'Bag', 'Hips', 'Torso']
        Bag = filenames[type]['Bag']['Label']
        Hips = filenames[type]['Hips']['Label']
        Torso = filenames[type]['Torso']['Label']
        #检查几个label是否一致
        df_Hand = read_label(Hand)
        df_Bag = read_label(Bag)
        df_Hips = read_label(Hips)
        df_Torso = read_label(Torso)

        if df_Hand.equals(df_Bag) and df_Hand.equals(df_Torso) and df_Hand.equals(df_Torso):
            print("All DataFrames are the same")
            print(type,'ok')
        else:
            print("DataFrames are not the same")
            print(type,'not ok')
    
def traj_fun(filename):
    train_label = pd.read_pickle(filename)
    train_label['index'] = train_label.index
    train_label['timestamp_diff'] = abs(train_label['timestamp'].diff()) >10
    train_label['trajectory_id'] = train_label['timestamp_diff'].cumsum()
    train_label = train_label.drop('timestamp_diff', axis=1)
    print(len(train_label['trajectory_id'].unique()))
    train_label.to_pickle(filename)
    return train_label

idx_start = 0
for data in ['train','valid','test']:
    print(filenames[data]['Label'])
    label = pd.read_pickle(filenames[data]['Label'].replace('.txt','.pkl'))
    label['idx'] = np.arange(len(label)) 
    label['idx'] = label['idx']+ idx_start
    idx_start += len(label['idx'])
    label.to_pickle(filenames[data]['Label'].replace('.txt','.pkl'))


label_idx_start = 0
for data in ['train','valid','test']:

    print(filenames[data]['Label'])
    label = pd.read_pickle(filenames[data]['Label'])
    if data =='test':
      label['label'] = -1000
    day = label['timestamp']//(1000 * 60 *60 * 24)
    diff =( abs(label['label'].diff())>0) | (abs(day.diff())>0 )
    label['label_idx'] = diff.cumsum() + label_idx_start
    label.to_pickle(filenames[data]['Label'])
    label_idx_start += len(label['label_idx'].unique())

'''  
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
G = ox.graph_from_bbox(50.737863484, 53.494297811, -2.980204386, 1.057061819, network_type='drive')
ox.save_graph_xml(G, filepath='network.osm')
    
'''
'''
idx_start = 0
for data in ['train','valid','test']:
    print(filenames[data]['Label'])
    label = pd.read_pickle(filenames[data]['Label'].replace('.txt','.pkl'))
    label['idx'] = label['idx']+ idx_start
    idx_start += len(label['idx'])
    label.to_pickle(filenames[data]['Label'].replace('.txt','.pkl'))
'''

'''
label = pd.read_pickle(filenames['test']['Label'].replace('.txt','.pkl'))
label['label'] = -1000
label.to_pickle(filenames['test']['Label'].replace('.txt','.pkl'))
'''
