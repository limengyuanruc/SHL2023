# Code that might generate warnings
# %%
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
import math
import time
import pandas as pd
import numpy as np
import progressbar
from joblib import Parallel, delayed
import multiprocessing
from scipy.stats import skew, kurtosis
from numpy.lib.stride_tricks import as_strided as stride
from geopy.distance import geodesic
import _pickle as cPickle

# 每一条轨迹长短不一，在轨迹内采用滑动窗口L1（即决定窗口）,步长为L2,划分为子轨迹，提取子轨迹的特征。
# 在子轨迹上训练模型，最后可以基于轨迹进行后处理
L1 = 5  # s 滑动窗口的长度，暂时依据是20年直接给的5s的数据
L2 = 1  # 滑动窗口的步长，假设1s内没有变化（主要原因是GPS数据是1s记录的）
row_num = 1000000  # 用于测试的数据量大小
num_cores = 8  # 调用cpu核心数
ll_1 = 10 * 60 * 100
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

'''
由于陀螺仪给的是手机三个方向的角速度，
所以乘对应的时间间隔便是这段时间内手机在对应方向上的角度旋转
'''

start_1 = time.time()
def process_data(dataset='valid', loc='Hand'):
    #print(dataset)
    datas = {}
    if dataset=='test':
        datas['sensors'] = pd.read_pickle(filenames[dataset]['Sensors'])
        datas['location'] = pd.read_pickle(filenames[dataset]['Location_new'])
        datas['location'][['num','satellite','snr','azimuth','elevation']] \
            = pd.read_pickle(filenames[dataset]['GPS_new'])[['num','satellite','snr','azimuth','elevation']]
    else:
    
        datas['sensors'] = pd.read_pickle(filenames[dataset][loc]['Sensors'])
        datas['location'] = pd.read_pickle(filenames[dataset][loc]['Location_new'])
        datas['location'][['num','satellite','snr','azimuth','elevation']] \
            = pd.read_pickle(filenames[dataset][loc]['GPS_new'])[['num','satellite','snr','azimuth','elevation']]
        #datas['stat'] = pd.read_pickle(filenames[dataset][loc]['Sensors'].replace('Sensors','data'))
    return datas
#datas= process_data(dataset='valid', loc='Hand')

#%%

def compute_distance(fLat, fLon, sLat, sLon):  # 计算距离

    return geodesic((fLat, fLon), (sLat, sLon)).meters


def feats_1(x): # sensors 的窗口特征
    """
    均值、方差、最大值、最小值、中位数、众数、
    均方根、标准差、振幅、偏度、峰度、零交率等
    """
    _mean = np.mean(x)
    _var = np.var(x)
    _max = np.max(x)
    _min = np.min(x)
    _median = np.median(x)
    _RMS = np.sqrt(np.mean(x ** 2))
    _std = np.std(x)
    _amp = np.abs(max(x) - min(x))
    _skew = skew(x)
    _kurtosis = kurtosis(x)
    _ZeroCrossRate = len(np.where(np.diff(x > 0))[0]) / len(x) if len(x) > 0 else 0

    return np.array([_mean, _var, _max, _min, _median, _RMS, _std, _amp, _skew, _kurtosis, _ZeroCrossRate])


def feats_2(x):  # 频域特征

    return np.array([np.sum(x ** 2), np.mean(x)])


def feats_rect(x):  # 计算L1s内 始终点的距离/L1 即平均速度

    if np.isnan(x).any():

        return np.array([np.nan])
    else:
        dist = compute_distance(x[0], x[1], x[2], x[3])

        return np.array([dist]) / L1


def feats_4(x):  # L1窗口内 速度、加速度的特征

    return np.array([np.nanmean(x), np.nanmedian(x)])


def feats_5(x):  # L1窗口内 卫星接受的一些特征
    y_mean = [np.nanmean(i) for i in x]
    y_max = [np.nan if i == list([]) else np.nanmax(i) for i in x]

    return np.array([np.nanmean(y_mean), np.nanmax(y_max)])


def feat_nan(x):  # L1窗口内 0点的比例

    return np.sum(np.isnan(x)) / len(x)


def get_stride(y, window, shift):  # 利用numpy stride函数 划分带步长的滑动窗口
    dim0, dim1 = y.shape
    stride0, stride1 = y.strides
    # stride0表示在沿dim0维度到下一个位置，跳过多少字节。
    stride_values = stride(y, shape=((dim0 - (window - shift)) // shift, window, dim1),
                           strides=(stride0 * shift, stride0, stride1))
    # shape 目标数组的shape，即   num = (dim0 - (window - shift)) // shift  个 window * dim1 的数组
    # strides 目标数组的strides

    return stride_values





def get_subindex(leng,ll_1):
    ranges = []
    ll_2 = ll_1//3
    start = 0
    while True:        
        end = min(start+ll_1, leng)
        ranges.append([int(start), int(end)])
        start += ll_2
        if end == leng:
            break        
    return ranges

#%%
def get_data(dataset=dataset, loc=loc):
    datas = process_data(dataset=dataset, loc=loc)
    bar = progressbar.ProgressBar()
    sub_sensors_matrices = []
    sub_gps_matrices = []#经纬度速度等
    
    sub_sensors_stat = []#经纬度速度等
    sub_gps_stat = []#经纬度速度等

    for i in bar(range(datas['sensors']['trajectory_id'].max()+1)):
        sub_sensors_matrices_i = []
        sub_gps_matrices_i = []#经纬度速度等
        # 由于每一条轨迹长短不一，在轨迹内采用窗口L1,步长为L2,划分为子轨迹，提取子轨迹的特征，然后进行后处理。
        label_i = datas['sensors'][datas['sensors']['trajectory_id'] == i].reset_index(drop=True)
        # 完整的秒
        indices = np.where(label_i['timestamp'] % 1000 == 0)[0] #所有的完整的s

        if len(indices)>0 : #如果大于5s ，这里是label_i的index
            if indices[-1]- indices[0] >  L1 * 100:
                label_i = label_i[(indices[0] + 1):(indices[-1] + 1)].reset_index(drop = True) #001 ~ 100
                #print(i, len(label_i))
                index_i = [label_i['idx'].min(), label_i['idx'].max()]  # 提取对应的idx相对应
                location_i = datas['location'][
                    (datas['location']['idx'] >= index_i[0]) & (datas['location']['idx'] <= index_i[1])].reset_index(drop = True)

                #然后我们 使用长度为600（*100）的窗口进行划分子轨迹，滑动步长是200，
                sub_index = get_subindex(len(label_i),ll_1)
                for p in sub_index:
                    if p[1]-p[0] > L1 * 100:
                        sub_sensors = label_i[p[0]:p[1]][['acc_total', 'gyr_total', 'mag_total','acc_x','acc_y','acc_z','gyr_x','gyr_y','gyr_z','mag_x','mag_y','mag_z']].values
                        sub_gps = location_i[p[0]//100:p[1]//100][['latitude', 'longitude', 'speed', 'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert','label','idx','trajectory_id','label_idx']].values
                        sub_sensors_matrices_i.append(sub_sensors)
                        sub_gps_matrices_i.append(np.nan_to_num(sub_gps))
                sub_sensors_matrices.append(sub_sensors_matrices_i)
                sub_gps_matrices.append(sub_gps_matrices_i)
    if dataset == 'test':
        cPickle.dump((sub_sensors_matrices,sub_gps_matrices), open('/DATA2/lvxiaoling/limengyuan/SHL2023/{}/raw_data.pkl'.format(dataset), "wb"))
    else:
        cPickle.dump((sub_sensors_matrices,sub_gps_matrices), open('/DATA2/lvxiaoling/limengyuan/SHL2023/{}/{}/raw_data.pkl'.format(dataset,loc), "wb"))

#%%
get_data(dataset='test', loc=loc)
'''

for dataset in ['valid','train']:
    for loc in ['Hand', 'Bag', 'Hips', 'Torso']:
        print(dataset, loc)
        get_data(dataset=dataset, loc=loc)


'''
