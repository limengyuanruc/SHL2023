# Code that might generate warnings
# %%
import warnings
#warnings.filterwarnings("ignore")  # Ignore all warnings
from tqdm import tqdm

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
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('my_logger')
# 每一条轨迹长短不一，在轨迹内采用滑动窗口L1（即决定窗口）,步长为L2,划分为子轨迹，提取子轨迹的特征。
# 在子轨迹上训练模型，最后可以基于轨迹进行后处理
L1 = 5  # s 滑动窗口的长度，暂时依据是20年直接给的5s的数据
L2 = 1  # 滑动窗口的步长，假设1s内没有变化（主要原因是GPS数据是1s记录的）
row_num = 1000000000  # 用于测试的数据量大小
num_cores = 8  # 调用cpu核心数

locs = ['Hand', 'Bag', 'Hips', 'Torso']

loc = locs[0]  # 'Hand'
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

def fun_mean_max(x):
    
    x = np.array(x['snr'])
    
    if np.isnan(x).all():
        return 0,0
    elif len(x)==0:
        return 0,0
    else:
        return np.mean(x),np.max(x)
# 为了验证代码的正确性，可以调整读取的数据集的大小
start_1 = time.time()
def process_data(dataset='valid', loc='Bag'):
    #print(dataset)
    if dataset == 'test':
        datas = {}
        
        datas['sensors'] = pd.read_pickle(filenames[dataset]['Sensors'])
        datas['location'] = pd.read_pickle(filenames[dataset]['Location_new'])
        datas['location'][['num','satellite','snr','azimuth','elevation']] \
            = pd.read_pickle(filenames[dataset]['GPS_new'])[['num','satellite','snr','azimuth','elevation']]
        datas['location'][['snr_mean', 'snr_max']] = datas['location'][['snr']].apply(fun_mean_max, axis=1, result_type='expand')
    else:
        datas = {}
        
        datas['sensors'] = pd.read_pickle(filenames[dataset][loc]['Sensors'])
        datas['location'] = pd.read_pickle(filenames[dataset][loc]['Location_new'])
        datas['location'][['num','satellite','snr','azimuth','elevation']] \
            = pd.read_pickle(filenames[dataset][loc]['GPS_new'])[['num','satellite','snr','azimuth','elevation']]
        datas['location'][['snr_mean', 'snr_max']] = datas['location'][['snr']].apply(fun_mean_max, axis=1, result_type='expand')
    
    return datas
#datas = process_data(dataset, loc)
'''
sensors:
['timestamp', 'label', 'time', 'idx', 'trajectory_id', 'acc_total',
       'gyr_total', 'mag_total', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y',
       'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'rotation_azimuth',
       'rotation_pitch', 'rotation_roll', 'acc1_x', 'acc1_y', 'acc1_z',
       'mag1_x', 'mag1_y', 'mag1_z']

location:
['timestamp', 'time', 'label', 'trajectory_id', 'idx', 'timestamp1',
       'accuracy', 'latitude', 'longitude', 'altitude', 'gps_trip_id', 'speed',
       'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert', 'time1',
       'index1', 'geometry', 'raliways_class', 'transport_class',
       'traffic_class', 'landuse_class', 'roads_class', 'roads_code', 'num',
       'satellite', 'snr', 'azimuth', 'elevation']

'''

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
def feats_geo(x):
    x = x[x>0]

    if len(x)== 0 :
        return 0
    else:
        return np.argmax(np.bincount(x))
    




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



#%%

# 并行计算
def get_r(datas):
    data_results = []
    for i in tqdm(range(datas['sensors']['trajectory_id'].max()+1)):
        # 由于每一条轨迹长短不一，在轨迹内采用窗口L1,步长为L2,划分为子轨迹，提取子轨迹的特征，然后进行后处理。
        label_i = datas['sensors'][datas['sensors']['trajectory_id'] == i].reset_index(drop=True)
        logger.info('{},{}'.format(i, len(label_i)))

        # 完整的秒
        indices = np.where(label_i['timestamp'] % 1000 == 0)[0]
        if len(indices) > 0:
            if indices[-1]- indices[0] >  L1 * 100:  # 如果持续时间大于L1s
                label_i = label_i[(indices[0] + 1):(indices[-1] + 1)].reset_index(drop=True)
                #####################
                #####处理 sensors 数据
                #####滑动窗口内有L1*100个点
                #####################
                target_columns = ['acc_x', 'acc_y', 'acc_z', 'acc_total',
                                'gyr_x', 'gyr_y', 'gyr_z', 'gyr_total',
                                'mag_x', 'mag_y', 'mag_z', 'mag_total',
                                'rotation_azimuth', 'rotation_pitch',
                                'rotation_roll']#

                # 时域特征
                data_sensors = label_i[target_columns].values
                data_sensors_stride = get_stride(data_sensors, L1 * 100, L2 * 100)  # 滑动窗口划分的序列 num * L1 * feat_num
                data_sensors_result = np.apply_along_axis(feats_1, axis=1, arr=data_sensors_stride)  # num * n * feat_num
                data_sensors_result = pd.DataFrame(data_sensors_result.reshape(data_sensors_result.shape[0], -1))
                data_sensors_result.columns = [j + i for i in
                                            ['_mean', '_var', '_max', '_min', '_median', '_RMS', '_std', '_amp', '_skew',
                                                '_kurtosis', '_ZeroCrossRate'] for j in target_columns]

                # 频域特征
                spectrum_columns = []
                for target_column in target_columns:
                    label_i['{}_spectrum'.format(target_column)] = np.abs(np.fft.fft(label_i[target_column]))
                    spectrum_columns.append('{}_spectrum'.format(target_column))

                data_spectrum = label_i[spectrum_columns].values
                data_spectrum_stride = get_stride(data_spectrum, L1 * 100, L2 * 100)  # 滑动窗口划分的序列 num * L1 * feat_num
                data_spectrum_result = np.apply_along_axis(feats_2, axis=1, arr=data_spectrum_stride)  # num * L1 * feat_num
                data_spectrum_result = pd.DataFrame(data_spectrum_result.reshape(data_spectrum_result.shape[0], -1))
                data_spectrum_result.columns = [j + i for i in ['_spectrum_energy', '_spectrum_average'] for j in
                                                target_columns]

                data_result = pd.concat([data_sensors_result, data_spectrum_result], axis=1)

                #####################
                #####处理GPS数据，首先把对应的GPS数据提取出来，只有L1个点（最多），可能都是nan
                #####################
                index_i = [label_i['idx'].min(), label_i['idx'].max()]  # 完整的s 不同文件中的idx相对应
                location_i = datas['location'][
                    (datas['location']['idx'] >= index_i[0]) & (datas['location']['idx'] <= index_i[1])].reset_index(drop=True)

                gps_loc_i = location_i[['latitude', 'longitude']].values
                gps_loc_i_stride = get_stride(gps_loc_i, L1, L2)  # num,L1,2

                # 缺失的比例
                data_result['location_nan_ratio'] = np.apply_along_axis(feat_nan, axis=1, arr=gps_loc_i_stride[:, :, 0])

                # 窗口内起始经纬度间的距离，如果存在缺失点则为nan
                gps_rect = np.concatenate([gps_loc_i_stride[:, 0, :], gps_loc_i_stride[:, -1, :]], axis=1)  # num,4
                rect_results = np.apply_along_axis(feats_rect, axis=1, arr=gps_rect)

                data_result['rect_dist'] = rect_results

                # 首先是该点本身的实时的特征
                location_columns = ['accuracy', 'altitude', 'speed', 'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']
                data_result[location_columns] = location_i[location_columns][L1 - 1:].values

                # 其次窗口内，一些时域时域的特征
                location_columns = ['speed', 'bearing', 'acc_lng', 'acc_lat', 'bearing_rate', 'jert']

                data_location = location_i[location_columns].values
                data_location_stride = get_stride(data_location, L1, L2)  # 滑动窗口划分的序列 num * L1 * feat_num
                data_location_result = np.apply_along_axis(feats_4, axis=1, arr=data_location_stride)  # num * L1 * feat_num
                data_location_result = pd.DataFrame(data_location_result.reshape(data_location_result.shape[0], -1))
                data_location_result.columns = [j + i for i in ['_mean', '_median'] for j in location_columns]
                data_result = pd.concat([data_result, data_location_result], axis=1)

                # 对路网信息进行处理
                geo_columns = ['raliways_class', 'transport_class','traffic_class', 'landuse_class', 'roads_class', 'roads_code']
                data_geo = location_i[geo_columns].values
                data_geo_stride = get_stride(data_geo, L1, L2) 
                data_geo_result = np.apply_along_axis(feats_geo, axis=1, arr=data_geo_stride)  # num * L1 * feat_num
                data_result[geo_columns] = data_geo_result.astype(int)

                

                # 缺失的比例
                gps_rec_i = location_i[['num']].values
                gps_rec_i_stride = get_stride(gps_rec_i, L1, L2)  # num,5,2
                # 缺失的比例
                data_result['reception_nan_ratio'] = np.apply_along_axis(feat_nan, axis=1,
                                                                        arr=gps_rec_i_stride.astype(float))
                data_result['reception_num'] = np.apply_along_axis(np.nanmean, axis=1,
                                                                arr=gps_rec_i_stride.astype(float))
                gps_snr_i = location_i[['snr_mean', 'snr_max']].values
                gps_snr_i_stride = get_stride(gps_snr_i, L1, L2) 
                data_result['snr_mean'] = np.apply_along_axis(np.mean, axis=1, arr=gps_snr_i_stride[:,:,0])
                data_result['snr_max'] = np.apply_along_axis(np.max, axis=1, arr=gps_snr_i_stride[:,:,1])

                data_result['label'] = label_i['label'][(L2*99)::(L2 * 100)][L1 - 1:].values  # 每隔L2个取一个label
                data_result['label_idx'] = label_i['label_idx'][(L2*99)::(L2 * 100)][L1 - 1:].values  # 每隔L2个取一个label
                data_result['idx'] = label_i['idx'][(L2*99)::(L2 * 100)][L1 - 1:].values
                data_result['timestamp'] = label_i['timestamp'][(L2*99)::(L2 * 100)][L1 - 1:].values
                data_result['trajectory_id'] = label_i['trajectory_id'].values[0]
        data_results.append(data_result)
    return pd.concat(data_results).reset_index(drop=True)
'''
for dataset in ['valid','train']:
    for loc in [ 'Hips', 'Torso','Hand', 'Bag',]:
        print(dataset, loc)
        start_2 = time.time()
        datas = process_data(dataset, loc)
        print('特征提取...')
        results = get_r(datas)#Parallel(n_jobs=4)(delayed(processInput_feature)(i) )
        results.to_pickle( '/DATA2/lvxiaoling/limengyuan/SHL2023/{}/{}/data.pkl'.format(dataset,loc))
        del results
        print('特征提取完成')
        end_2 = time.time()
        duration_2 = end_2 - start_2
        print(f"特征提取用时: {int(duration_2)}秒")
'''      
dataset = 'test'
print(dataset)
datas = process_data(dataset)
start_2 = time.time()
results = get_r(datas)
results.to_pickle( '/DATA2/lvxiaoling/limengyuan/SHL2023/{}/data.pkl'.format(dataset))
end_2 = time.time()
duration_2 = end_2 - start_2
print(f"特征提取用时: {int(duration_2)}秒")

#%%
