# Code that might generate warnings
# %%
from pandarallel import pandarallel

import warnings
#warnings.filterwarnings("ignore")  # Ignore all warnings
import math
import time
import json
import pandas as pd
import numpy as np
import progressbar
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import skew, kurtosis
from numpy.lib.stride_tricks import as_strided as stride
from geopy.distance import geodesic
from tqdm import tqdm
import multiprocessing

# 每一条轨迹长短不一，在轨迹内采用滑动窗口L1（即决定窗口）,步长为L2,划分为子轨迹，提取子轨迹的特征。
# 在子轨迹上训练模型，最后可以基于轨迹进行后处理
L1 = 5  # s 滑动窗口的长度，暂时依据是20年直接给的5s的数据
L2 = 1  # 滑动窗口的步长，假设1s内没有变化（主要原因是GPS数据是1s记录的）
row_num = 10000000000  # 用于测试的数据量大小
num_cores = 8  # 调用cpu核心数


locs = ['Hand', 'Bag', 'Hips', 'Torso']

loc = locs[0]  # 'Hand'
dataset = 'train'

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
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/sensors.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hand/GPS_new.pkl',
        },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/Acc.pkl',
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/sensors.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Bag/GPS_new.pkl',
        },
        'Hips': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/Acc.pkl',
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/sensors.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Hips/GPS_new.pkl',
        },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/Acc.pkl',
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/train/Torso/sensors.pkl',
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
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/sensors.pkl',
            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hand/GPS_new.pkl',
        },
        'Bag': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/Gyr.pkl',
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Bag/sensors.pkl',

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
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/sensors.pkl',

            'GPS': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS.pkl',
            'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Hips/GPS_new.pkl',
        },
        'Torso': {
            'Location': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location.pkl',
            'Location_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Location_new.pkl',
            'Mag': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Mag.pkl',
            'Gyr': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Gyr.pkl',
            'Acc': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/Acc.pkl',
            'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/valid/Torso/sensors.pkl',

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
        'Sensors': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/sensors.pkl',
        'GPS_new': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/GPS_new.pkl',
        'Label': '/DATA2/lvxiaoling/limengyuan/SHL2023/test/Label.pkl'
    }
}

# 方位计的计算
'''
方位计的计算仅需要用到加速度计以及磁力计，
也即 acc_x, acc_y, acc_z, mag_x, mag_y, mag_z 共6轴

代码使用安卓官方文档中的类方法 get_rotationMatrix 和 get_orientation
其中 get_rotationMatrix 可以得到 get_orientation 的参数，最后由 get_orientation 计算出方位
'''
def coordinate_transformation(data):
    """
    Ax, Ay, Az: 加速度计三轴数据，重力计和加速度计在计算方位时可以认为是等价的
    Ex, Ey, Ez: 磁力计三轴数据

    R: 最后的输出结果，将用于 get_orientation 函数
    """
    
    Ax = data['acc_x']
    Ay = data['acc_y']
    Az = data['acc_z']
    Ex = data['mag_x']
    Ey = data['mag_y']
    Ez = data['mag_z']
    AD = [Ax,Ay,Az]
    ED = [Ex,Ey,Ez]
    

    R = [0] * 9  # 初始化
    

    Hx = Ey * Az - Ez * Ay  # 磁感应和重力叉乘得到水平向西的方向，在手机坐标系下的坐标值
    
    Hy = Ez * Ax - Ex * Az
    Hz = Ex * Ay - Ey * Ax
    
    normH = np.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)  # 用于归一化
    #print(normH)
    if normH < 0.1:
        # 手机做自由落体运动或者指向正北
        return R

    invH = 1.0 / normH
    Hx *= invH
    Hy *= invH  # 水平方向坐标值的归一化
    Hz *= invH
    invA = 1.0 / np.sqrt(Ax * Ax + Ay * Ay + Az * Az)
    Ax *= invA
    Ay *= invA  # 重力方向归一化
    Az *= invA
    Mx = Ay * Hz - Az * Hy
    My = Az * Hx - Ax * Hz  # 用归一化的重力和水平向西方向叉乘得到归一化的正北方向与地球相切（为什么不直接使用磁感应计方向呢?）
    Mz = Ax * Hy - Ay * Hx

    R[0] = Hx
    R[1] = Hy
    R[2] = Hz  # 得到旋转矩阵，x:水平向西的手机坐标表示,而且是归一化数值
    R[3] = Mx
    R[4] = My
    R[5] = Mz  # y:由南向北风向
    R[6] = Ax
    R[7] = Ay
    R[8] = Az  # 重力轴方向
    acc1_x, acc1_y, acc1_z = get_newcoordinate(AD,R)
    mag1_x, mag1_y, mag1_z = get_newcoordinate(ED,R)

    rotation_azimuth, rotation_pitch, rotation_roll = get_orientation(R)
    return rotation_azimuth, rotation_pitch, rotation_roll ,acc1_x, acc1_y, acc1_z,mag1_x, mag1_y, mag1_z
'''
https://stackoverflow.com/questions/15315129/convert-magnetic-field-x-y-z-values-from-device-into-global-reference-frame
A_W[0] = R[0] * A_D[0] + R[1] * A_D[1] + R[2] * A_D[2];
A_W[1] = R[3] * A_D[0] + R[4] * A_D[1] + R[5] * A_D[2];
A_W[2] = R[6] * A_D[0] + R[7] * A_D[1] + R[8] * A_D[2];
'''
def  get_newcoordinate(A_D,R):
    A_W_0 = R[0] * A_D[0] + R[1] * A_D[1] + R[2] * A_D[2]
    A_W_1 = R[3] * A_D[0] + R[4] * A_D[1] + R[5] * A_D[2]
    A_W_2 = R[6] * A_D[0] + R[7] * A_D[1] + R[8] * A_D[2]
    return A_W_0,A_W_1,A_W_2



def get_orientation(R):
    """
    R: get_rotationMatrix 函数的输出

    values: 最后得到的方位，三个值分别为航向角、俯仰角和翻滚角
    """
    rotation_azimuth = math.atan2(R[1], R[4])  # azimuth 航向角 yaw
    rotation_pitch = math.asin(-R[7])  # pitch 俯仰角
    rotation_roll = math.atan2(-R[6], R[8])  # roll 横滚角

    return rotation_azimuth, rotation_pitch, rotation_roll




# 陀螺仪数据的使用
'''
由于陀螺仪给的是手机三个方向的角速度，
所以乘对应的时间间隔便是这段时间内手机在对应方向上的角度旋转

希望返回的是 ['acc_x', 'acc_y', 'acc_z', 'acc_total',
                          'gyr_x', 'gyr_y', 'gyr_z', 'gyr_total',
                          'mag_x', 'mag_y', 'mag_z', 'mag_total',

                          'acc1_x', 'acc1_y', 'acc1_z',
                          'mag1_x', 'mag1_y', 'mag1_z', 

                          'rotation_azimuth', 'rotation_pitch',
                          'rotation_roll']

'''
# 为了验证代码的正确性，可以调整读取的数据集的大小
#%%
def process_data(dataset='train', loc='Hand'):
        
    sensors_data = pd.read_pickle(filenames[dataset]['Label'])
    print(dataset, loc)
    if dataset == 'test':
        acc = pd.read_pickle(filenames[dataset]['Acc']).interpolate()
        gyr = pd.read_pickle(filenames[dataset]['Gyr']).interpolate()
        mag = pd.read_pickle(filenames[dataset]['Mag']).interpolate()
        save_dir = '/DATA2/lvxiaoling/limengyuan/SHL2023/{}/sensors.pkl'.format(dataset)
    else:
        acc = pd.read_pickle(filenames[dataset][loc]['Acc']).interpolate()
        gyr = pd.read_pickle(filenames[dataset][loc]['Gyr']).interpolate()
        mag = pd.read_pickle(filenames[dataset][loc]['Mag']).interpolate()
        save_dir = '/DATA2/lvxiaoling/limengyuan/SHL2023/{}/{}/sensors.pkl'.format(dataset,loc)


    acc_x = acc['acc_x']
    acc_y = acc['acc_y']
    acc_z = acc['acc_z']
    sensors_data['acc_total'] = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    
    
    gyr_x = gyr['gyr_x']
    gyr_y = gyr['gyr_y']
    gyr_z = gyr['gyr_z']
    sensors_data['gyr_total'] = np.sqrt(gyr_x ** 2 + gyr_y ** 2 + gyr_z ** 2)


    mag_x = mag['mag_x']
    mag_y = mag['mag_y']
    mag_z = mag['mag_z']
    sensors_data['mag_total'] = np.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)

    sensors_data[['acc_x','acc_y','acc_z']] = acc[['acc_x','acc_y','acc_z']]
    sensors_data[['gyr_x','gyr_y','gyr_z']] = gyr[['gyr_x','gyr_y','gyr_z']]
    sensors_data[['mag_x','mag_y','mag_z']] = mag[['mag_x','mag_y','mag_z']]

    tqdm.pandas()
    pandarallel.initialize(progress_bar=True,nb_workers=30)

    sensors_data[['rotation_azimuth', 'rotation_pitch', 'rotation_roll', 
                  'acc1_x', 'acc1_y', 'acc1_z',
                    'mag1_x', 'mag1_y', 'mag1_z' ]] \
        = sensors_data[['acc_x','acc_y','acc_z','mag_x','mag_y','mag_z']].parallel_apply(coordinate_transformation, axis=1, result_type='expand')
    print('saving')
    sensors_data.to_pickle(save_dir)
    return sensors_data

for dataset in ['valid','train']:
    for loc in [ 'Hips', 'Torso','Hand', 'Bag',]:
        print(dataset, loc)
        process_data(dataset=dataset, loc=loc)
dataset = 'test'
process_data(dataset=dataset, loc=loc)


'''
#%%
#增加一列label id
label_idx_start = 0
for data in ['train','valid','test']:
    print(filenames[data]['Label'])
    label = pd.read_pickle(filenames[data]['Label'])
    day = label['timestamp']//(1000 * 60 *60 * 24)
    diff =( abs(label['label'].diff())>0) | (abs(day.diff())>0 )
    label['label_idx'] = diff.cumsum() + label_idx_start
    label.to_pickle(filenames[data]['Label'])
    label_idx_start += len(label['label_idx'].unique())


for data in ['valid','train']:
    label = pd.read_pickle(filenames[data]['Label'])
    for loc in [ 'Hips', 'Torso','Hand', 'Bag',]:
        print(data,loc)
        sensors = pd.read_pickle(filenames[data][loc]['Sensors'])
        sensors['label_idx'] = label['label_idx']
        sensors.to_pickle(filenames[data][loc]['Sensors'])


data = 'test'
label = pd.read_pickle(filenames[data]['Label'])
sensors = pd.read_pickle(filenames[data]['Sensors'])
sensors['label_idx'] = label['label_idx']
sensors.to_pickle(filenames[data]['Sensors'])

'''