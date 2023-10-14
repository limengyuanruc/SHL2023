# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.functional as f
from torch.nn import Parameter
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence 
import itertools
from sklearn.metrics import f1_score


from prettytable import PrettyTable
import datetime

import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

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
import argparse
import random
from utils import *
from sklearn.model_selection import train_test_split
print('lmy/7_model_gps_road.py')

# %% [markdown]
# # 在训练集上训练

# %%
filenames =get_filenames()

# %%
parser = argparse.ArgumentParser()
#关于IB模型的参数
parser.add_argument('--type', type=str, default='lstm')#这是啥 大于0的参与训练

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--L1', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weightloss', type=float, default=0)

parser.add_argument('--transformer', type=int, default=1)#这是啥 大于0的参与训练



parser.add_argument('--num_filter', type=int, default=64)
parser.add_argument('--STAT_NET_road_embedding', type=int, default=16)
parser.add_argument('--STAT_NET_sensors_embedding', type=int, default=64)
parser.add_argument('--STAT_NET_geo_embedding', type=int, default=64)

parser.add_argument('--mix', type=int, default=0)#这是啥 大于0的参与训练



# Bi-LSTM parser
parser.add_argument('--lstm_layer', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--epochs', type=int, default=400)


parser.add_argument('--save_dir', type=str, default='save0528')
parser.add_argument('--seed', type=int, default=42)#这是啥 大于0的参与训练
parser.add_argument('--nheads', type=int, default=3)#这是啥 大于0的参与训练
parser.add_argument('--n_layers', type=int, default=2)#这是啥 大于0的参与训练
parser.add_argument('--ff_size', type=int, default=128)#这是啥 大于0的参与训练





parser.add_argument('--lr', type=float, default=0.001)


parser.add_argument('--pad_value', type=int, default=-1000) 
parser.add_argument('--num_workers', type=int, default=4) 



args = parser.parse_args()
args.STAT_NET_input_road = np.array([ 8, 11, 20, 21, 27,  6])#train_data.stat_data.loc[:,get_road_name()].max().values+1#最大值加1


args.device = 'cuda:{}'.format(args.gpu) if (args.gpu>=0) & torch.cuda.is_available() else 'cpu'
args.now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f==') #时间
args.model_save = '/root/autodl-tmp/save/models{}.pth'.format(args.now)
print(args.model_save)

# %%
def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
seed_torch(args.seed)

# %%
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

# %%
class  dset(Dataset):
    def __init__(self, data_sensor,data_gps,stat_data):
        super(dset, self).__init__()
        self.data_sensor = data_sensor
        self.data_gps = data_gps
        self.stat_data = stat_data
        self.name_dict = {name:i for i,name in enumerate(stat_data)}
        self.stat_sensors = [self.name_dict[i] for i in get_sensors_name()]
        self.stat_gps = [self.name_dict[i] for i in get_gps_name()]
        self.stat_road = [self.name_dict[i] for i in get_road_name()]
    def __len__(self):
        return len(self.data_gps)
    def  __getitem__(self, i):
        stat_value = self.stat_data[self.stat_data['idx'].isin(self.data_gps[i][args.L1-1:,-3])].values
        if len(stat_value)!= len(self.data_gps[i][args.L1-1:,-3]):
            print(i)
        return torch.tensor(self.data_sensor[i]),torch.tensor(self.data_gps[i]),torch.tensor(stat_value[:,self.stat_sensors]),torch.tensor(stat_value[:,self.stat_gps]),torch.tensor(stat_value[:,self.stat_road]).long()
    #返回的10分钟内的数据
    #data_sensor,data_gps,label,idx

# %% [markdown]
# ## 加载数据

# %%
def collate_batch(batch):
    data_sensor_list,data_gps_list,lengths,stat_list1,stat_list2,stat_list3 =  [], [],[],[], [],[]
    batch.sort(key=lambda x: len(x[0]), reverse=True)#按照长度的大小进行排序
    for (data_sensor_,data_gps_,stat_list1_i,stat_list2_i,stat_list3_i) in batch:



        data_sensor_list.append(data_sensor_)
        data_gps_list.append(data_gps_)
        
        lengths.append(data_gps_.shape[0])
        stat_list1.append(stat_list1_i)
        stat_list2.append(stat_list2_i)
        stat_list3.append(stat_list3_i)

    data_sensor_list = pad_sequence(data_sensor_list, padding_value=args.pad_value, batch_first=True)#进行填充，每个batch中的句子需要有相同的长度
    data_gps_list = pad_sequence(data_gps_list, padding_value=args.pad_value, batch_first=True)#进行填充，每个batch中的句子需要有相同的长度
    
    stat_list1 = pad_sequence(stat_list1, padding_value=args.pad_value, batch_first=True)#进行填充，每个batch中的句子需要有相同的长度
    stat_list2 = pad_sequence(stat_list2, padding_value=args.pad_value, batch_first=True)#进行填充，每个batch中的句子需要有相同的长度
    stat_list3 = pad_sequence(stat_list3, padding_value=args.pad_value, batch_first=True)#进行填充，每个batch中的句子需要有相同的长度
    roads = []
    for i in range(6):
        road_i = stat_list3[:,:,[i]]
        road_i[road_i==args.pad_value] = args.STAT_NET_input_road[i]
        roads.append(road_i)
    stat_list3 = torch.cat(roads,dim=-1)

        #stat_list3[:,:,i][stat_list3[:,:,i]==args.pad_value] = args.STAT_NET_input_road[i]

    label =  torch.tensor(data_gps_list)[:,:,-4].long()
    label[label>0] = label[label>0]-1
    #data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, lengths in train_dataloader
    return data_sensor_list.float(),data_gps_list[:,:,:-4].float(),\
        stat_list1.float(),stat_list2.float(),stat_list3.long(),\
        label,torch.tensor(data_gps_list)[:,:,-3].long(),torch.tensor(data_gps_list)[:,:,-2].long(), lengths
        #'label','idx','trajectory_id','label_idx'\


# %%


# %%
#读入数据
print('loading data')
def dataset(args,loc='Hand'):
    print('raw data')

    train_hand = cPickle.load(open('/root/autodl-tmp/train/{}/raw_data.pkl'.format(loc),'rb'))
    print('data')
    stat_data = cPickle.load(open('/root/autodl-tmp/train/{}/data.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')


    data_x = np.arange(len(train_hand[0]))#trajectory id
    data_y = [int(i[0][0,-4])  for i in train_hand[1]]
    #划分训练集和测试集
    train_x,val_x,train_y,val_y = train_test_split(data_x,data_y,test_size=0.15,stratify=data_y,shuffle=True) #按y比例分层抽样，通过用于分类问题
#    train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.25,stratify=train_y,shuffle=True) #按y比例分层抽样，通过用于分类问题

    train_sensor = [j for i in train_x for j in train_hand[0][i]]
    train_gps = [j for i in train_x for j in train_hand[1][i]]
    val_sensor = [j for i in val_x for j in train_hand[0][i]]
    val_gps = [j for i in val_x for j in train_hand[1][i]]
#    test_sensor = [j for i in test_x for j in train_hand[0][i]]
#    test_gps = [j for i in test_x for j in train_hand[1][i]]

    train_data = dset(train_sensor,train_gps,stat_data)
    val_data = dset(val_sensor,val_gps,stat_data)
#    test_data = dset(test_sensor,test_gps,stat_data)


    args.CONV_SENSORS_input_dim = train_data.data_sensor[0].shape[1]
    args.CONV_GEO_num_feat =train_data.data_gps[0].shape[1]-4 - 2
    args.STAT_NET_input_sensors = train_data.stat_data.loc[:,get_sensors_name()].shape[1]
    args.STAT_NET_input_geo = train_data.stat_data.loc[:,get_gps_name()].shape[1]
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers = args.num_workers
        )


    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers = args.num_workers
        )

    '''
    train_hand = cPickle.load(open('/root/autodl-tmp/valid/{}/raw_data.pkl'.format(loc),'rb'))
    stat_data = cPickle.load(open('/root/autodl-tmp/valid/{}/data.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')   

    data_x = np.arange(len(train_hand[0]))#trajectory id
    data_y = [int(i[0][0,-4]) for i in train_hand[1]]
    val_sensor = [j for i in data_x for j in train_hand[0][i]]
    val_gps = [j for i in data_x for j in train_hand[1][i]]
    val_data = dset(val_sensor,val_gps,stat_data)


    test_loader1 = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers = args.num_workers
        )

    train_hand = cPickle.load(open('/root/autodl-tmp/test/raw_data.pkl','rb').format(loc))
    stat_data = cPickle.load(open('/root/autodl-tmp/test/data.pkl','rb').format(loc)).fillna(0).drop_duplicates(keep='first')   

    data_x = np.arange(len(train_hand[0]))#trajectory id
    data_y = [int(i[0][0,-4]) for i in train_hand[1]]
    val_sensor = [j for i in data_x for j in train_hand[0][i]]
    val_gps = [j for i in data_x for j in train_hand[1][i]]
    val_data = dset(val_sensor,val_gps,stat_data)


    test_loader2 = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers = args.num_workers
        )
    '''      
    return train_loader,val_loader,args
if args.mix:
    def dataset(args,loc='Hand'):
        print('raw data')

        train_hand = cPickle.load(open('/root/autodl-tmp/train/{}/raw_data_m.pkl'.format(loc),'rb'))
        print('data')
        stat_data = cPickle.load(open('/root/autodl-tmp/train/{}/data_m.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')


        data_x = np.arange(len(train_hand[0]))#trajectory id
        data_y = [int(i[0][0,-4])  for i in train_hand[1]]
        #划分训练集和测试集
        train_x,val_x,train_y,val_y = train_test_split(data_x,data_y,test_size=0.15,stratify=data_y,shuffle=True) #按y比例分层抽样，通过用于分类问题
    #    train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.25,stratify=train_y,shuffle=True) #按y比例分层抽样，通过用于分类问题

        train_sensor = [j for i in train_x for j in train_hand[0][i]]
        train_gps = [j for i in train_x for j in train_hand[1][i]]
        val_sensor = [j for i in val_x for j in train_hand[0][i]]
        val_gps = [j for i in val_x for j in train_hand[1][i]]
    #    test_sensor = [j for i in test_x for j in train_hand[0][i]]
    #    test_gps = [j for i in test_x for j in train_hand[1][i]]

        train_data = dset(train_sensor,train_gps,stat_data)
        val_data = dset(val_sensor,val_gps,stat_data)
    #    test_data = dset(test_sensor,test_gps,stat_data)


        args.CONV_SENSORS_input_dim = train_data.data_sensor[0].shape[1]
        args.CONV_GEO_num_feat =train_data.data_gps[0].shape[1]-4 - 2
        args.STAT_NET_input_sensors = train_data.stat_data.loc[:,get_sensors_name()].shape[1]
        args.STAT_NET_input_geo = train_data.stat_data.loc[:,get_gps_name()].shape[1]
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )


        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )

        '''
        train_hand = cPickle.load(open('/root/autodl-tmp/valid/{}/raw_data.pkl'.format(loc),'rb'))
        stat_data = cPickle.load(open('/root/autodl-tmp/valid/{}/data.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')   

        data_x = np.arange(len(train_hand[0]))#trajectory id
        data_y = [int(i[0][0,-4]) for i in train_hand[1]]
        val_sensor = [j for i in data_x for j in train_hand[0][i]]
        val_gps = [j for i in data_x for j in train_hand[1][i]]
        val_data = dset(val_sensor,val_gps,stat_data)


        test_loader1 = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )

        train_hand = cPickle.load(open('/root/autodl-tmp/test/raw_data.pkl','rb').format(loc))
        stat_data = cPickle.load(open('/root/autodl-tmp/test/data.pkl','rb').format(loc)).fillna(0).drop_duplicates(keep='first')   

        data_x = np.arange(len(train_hand[0]))#trajectory id
        data_y = [int(i[0][0,-4]) for i in train_hand[1]]
        val_sensor = [j for i in data_x for j in train_hand[0][i]]
        val_gps = [j for i in data_x for j in train_hand[1][i]]
        val_data = dset(val_sensor,val_gps,stat_data)


        test_loader2 = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )
        '''      
        return train_loader,val_loader,args
train_loader,val_loader,args = dataset(args)
print(args)
# %% [markdown]
# ## 模型

# %%
class CONV_SENSORS(nn.Module):
    def __init__(self,input_dim=3, num_filter = 64,kernel_size = 500, stride=100):
        super(CONV_SENSORS,self).__init__()
        self.conv = nn.Conv1d(input_dim, num_filter, kernel_size, stride = stride)

    def forward(self,data_sensor):# traj:batch_size*seq_len*17
        # 地理卷积
        data_sensor = data_sensor.permute(0,2,1)#batch_size,seq_len,num
        data_sensor = F.elu(self.conv(data_sensor)).permute(0,2,1)# L*seq_len'*num_filter
        return data_sensor


# %%
class CONV_GEO(nn.Module):
    def __init__(self,kernel_size=5,num_filter=64,num_feat = 6):
        super(CONV_GEO,self).__init__()
        self.process_coords = nn.Linear(2,16)
        self.conv1 = nn.Conv1d(16,num_filter,kernel_size)
        self.conv2 = nn.Conv1d(num_feat,num_filter,kernel_size)

    def forward(self,data_gps):# traj:batch_size*seq_len*17
        # 地理卷积
        
        lngs_lats = data_gps[:,:,:2] #batch_size*seq_len*2
        locs1 = torch.tanh(self.process_coords(lngs_lats))# batch_size*seq_len*16
        locs1 =locs1.permute(0,2,1)# batch_size*16*seq_len
        conv_locs1 = F.elu(self.conv1(locs1)).permute(0,2,1)# L*seq_len'*num_filter
        
        # 特征卷积
        features = data_gps[:,:,2:]# batch_size*seq_len*14
        locs2 = features.permute(0,2,1)# batch_size*14*seq_len
        conv_locs2 = F.elu(self.conv2(locs2)).permute(0,2,1)# L*seq_len'*num_filter
        
        return torch.concat([conv_locs1,conv_locs2],dim=2)#地理、特征、时间
        ## L*seq_len'*num_filter

# %%
class STAT_NET(nn.Module):
    def __init__(self,args=args,
                 input_road = [3,4,5,6,7,8],road_embedding =16, 
                 input_sensors=125,sensors_embedding = 64,
                 input_geo=125,geo_embedding = 64):
        super(STAT_NET, self).__init__()
        self.pad_value = args.pad_value
        self.args = args
        self.input_road = input_road
        self.emb = nn.ModuleList([nn.Embedding(i+1,road_embedding,padding_idx=i)    for  i in input_road])
        self.fc_sensors = nn.Linear(input_sensors,sensors_embedding)
        self.fc_geo = nn.Linear(input_geo,geo_embedding)
        #embedding层
        

    def forward(self, stat_sensors,stat_gps,stat_road):
        stat_sensors = self.fc_sensors(stat_sensors)
        stat_gps = self.fc_geo(stat_gps)

        roads = []
        for i,layer in enumerate(self.emb):
            road_i = stat_road[:,:,i]#batch_size,sqe_len,feat_num
            #road_i[road_i==args.pad_value] = self.input_road[i]
            road_i = layer(road_i)
            roads.append(road_i)
        roads = torch.cat(roads,dim=-1)
        return torch.cat([stat_gps,roads],dim=-1)

# %%
class BILSTM(torch.nn.Module):
    def __init__(self,args=args,input_dim=64+128, d_model = 128,out_dim=8):
        super(BILSTM, self).__init__()
        self.pad_value = args.pad_value
        self.args = args
        #embedding层
        self.lstm = nn.LSTM(input_dim,d_model//2, num_layers = args.lstm_layer, bidirectional = True,
                                dropout=args.dropout, batch_first=True)
        self.projection = nn.Linear(d_model, out_dim)

    def forward(self, x, lengths):
        lengths = torch.tensor(lengths)-(self.args.L1-1)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)#lstm层
        # hidden = [n layers *2, batch size, hidden dim]最后一个step的hidden
        # cell = [n layers * 2, batch size, hidden dim]最终一个step的cell
        x, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        x = self.projection(x)
        return x
if args.transformer:
    class BILSTM(torch.nn.Module):
        def __init__(self,args=args,input_dim=64+128, d_model = 128,out_dim=8):
            super(BILSTM, self).__init__()
            self.pad_value = args.pad_value
            self.args = args
            #embedding层
            self.lstm = nn.LSTM(input_dim,d_model//2, num_layers = args.lstm_layer, bidirectional = True,
                                    dropout=args.dropout, batch_first=True)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model * args.nheads, args.nheads, args.ff_size, args.dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, args.n_layers)
            self.fc = nn.Linear(d_model * args.nheads, d_model)

            self.projection = nn.Linear(d_model, out_dim)

        def forward(self, x, lengths):
            lengths = torch.tensor(lengths)-(self.args.L1-1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)        
            packed_output, (hidden, cell) = self.lstm(packed_embedded)#lstm层
            # hidden = [n layers *2, batch size, hidden dim]最后一个step的hidden
            # cell = [n layers * 2, batch size, hidden dim]最终一个step的cell
            x, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            mask = self.get_mask(lengths,x.shape[0])
            x = self.encoder(x.repeat(1, 1, self.args.nheads), src_key_padding_mask=mask)
            x = self.fc(x)
            x = self.projection(x)


            return x
        def get_mask(self,sentence_lengths,batch_len):
            # 计算最大句子长度
            max_length = max(sentence_lengths)

            # 创建一个零填充的张量，大小为(batch_size, max_length)
            src_key_padding_mask = torch.zeros((batch_len, max_length), dtype=torch.bool)

            # 对每个句子进行遍历，根据句子长度进行填充
            for i, length in enumerate(sentence_lengths):
                src_key_padding_mask[i, :length] = 1

# %%
if False:
    data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, length = next(iter(train_loader))

    m1 = CONV_SENSORS(input_dim=args.CONV_SENSORS_input_dim, num_filter =args.num_filter)
    r1 = m1(data_sensor)
    m2 = CONV_GEO(num_feat = args.CONV_GEO_num_feat,num_filter =args.num_filter)
    r2 = m2(data_gps)
    m3 = STAT_NET(  input_road = args.STAT_NET_input_road,road_embedding =args.STAT_NET_road_embedding, 
                    input_sensors=args.STAT_NET_input_sensors,sensors_embedding = args.STAT_NET_sensors_embedding,
                    input_geo=args.STAT_NET_input_geo,geo_embedding = args.STAT_NET_geo_embedding)
    r3 = m3(stat_sensors,stat_gps,stat_road)
    r = torch.cat([r1,r2,r3],dim=-1)

    m3 = BILSTM(input_dim=3*args.num_filter +args.STAT_NET_geo_embedding+len( args.STAT_NET_input_road)*args.STAT_NET_road_embedding,args=args)
    r4 = m3(r,length)




# %% [markdown]
# data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, length = next(iter(train_loader))
# 
# m1 = CONV_SENSORS(input_dim=args.CONV_SENSORS_input_dim, num_filter =args.num_filter)
# r1 = m1(data_sensor)
# m2 = CONV_GEO(num_feat = args.CONV_GEO_num_feat,num_filter =args.num_filter)
# r2 = m2(data_gps)
# m3 = STAT_NET(  input_road = args.STAT_NET_input_road,road_embedding =args.STAT_NET_road_embedding, 
#                  input_sensors=args.STAT_NET_input_sensors,sensors_embedding = args.STAT_NET_sensors_embedding,
#                  input_geo=args.STAT_NET_input_geo,geo_embedding = args.STAT_NET_geo_embedding)
# r3 = m3(stat_sensors,stat_gps,stat_road)
# r = torch.cat([r1,r2,r3],dim=-1)
# 
# m3 = BILSTM(input_dim=3*args.num_filter + args.STAT_NET_sensors_embedding+args.STAT_NET_geo_embedding+len( args.STAT_NET_input_road)*args.STAT_NET_road_embedding,args=args)
# r4 = m3(r,length)
# 
# 

# %% [markdown]
# ## 训练

# %%
def train_epoch(model_s,model_g,model_stat,model, optimizer,loss_fn,train_dataloader):
    model_s.train()
    model_g.train()
    model_stat.train()
    model.train()
    
    losses = 0
    correct = 0

    for data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, lengths in train_dataloader:
        data_sensor = data_sensor.to(args.device)
        data_gps = data_gps.to(args.device)
        stat_sensors,stat_gps,stat_road = stat_sensors.to(args.device),stat_gps.to(args.device),stat_road.to(args.device)
        label = label.to(args.device)[:,(args.L1-1):]

        output_s = model_s(data_sensor)
        output_g = model_g(data_gps)
        output_stat = model_stat(stat_sensors,stat_gps,stat_road)
        out = model(torch.cat([output_s,output_g,output_stat],dim=-1),lengths)

        optimizer.zero_grad()
        
        mask = label ==args.pad_value
        label = label

        loss = loss_fn( out.reshape(-1, out.shape[-1]), label.reshape(-1))
        loss.backward()

        optimizer.step()


        losses += loss.item() 
        
        pred = torch.argmax(out, dim=2)
        correct += (pred ==label ).masked_fill(mask,0).sum().item() / (~mask).sum()
        

    return losses / len(train_dataloader),correct/len(train_dataloader)

def evaluate(model_s,model_g,model_stat,model,loss_fn,train_dataloader):
    model_s.eval()
    model_g.eval()
    model_stat.eval()
    model.eval()

    
    losses = 0
    correct = 0
    with torch.no_grad():
        for data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, lengths in train_dataloader:
            data_sensor = data_sensor.to(args.device)
            data_gps = data_gps.to(args.device)
            stat_sensors,stat_gps,stat_road = stat_sensors.to(args.device),stat_gps.to(args.device),stat_road.to(args.device)
            label = label.to(args.device)[:,(args.L1-1):]

            output_s = model_s(data_sensor)
            output_g = model_g(data_gps)
            output_stat = model_stat(stat_sensors,stat_gps,stat_road)
            out = model(torch.cat([output_s,output_g,output_stat],dim=-1),lengths)

            
            loss = loss_fn( out.reshape(-1, out.shape[-1]), label.reshape(-1))
           
            losses += loss.item() 
            
            mask = label ==args.pad_value
            pred = torch.argmax(out, dim=2)
            correct +=  (pred ==label ).masked_fill(mask,0).sum().item() / (~mask).sum()
            
    model_s.train()
    model_g.train()
    model_stat.train()
    model.train()
    return losses / len(train_dataloader),correct/len(train_dataloader)

# %%
def weight_init(m):
    if isinstance(m, nn.Linear):
        
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# %%


# %%
model_s = CONV_SENSORS(input_dim=args.CONV_SENSORS_input_dim, num_filter =args.num_filter).to(args.device).apply(weight_init)
model_g = CONV_GEO(num_feat = args.CONV_GEO_num_feat,num_filter =args.num_filter).to(args.device).apply(weight_init)
model_stat = STAT_NET(input_road = args.STAT_NET_input_road,road_embedding =args.STAT_NET_road_embedding, 
                 input_sensors=args.STAT_NET_input_sensors,sensors_embedding = args.STAT_NET_sensors_embedding,
                 input_geo=args.STAT_NET_input_geo,geo_embedding = args.STAT_NET_geo_embedding).to(args.device).apply(weight_init)
model = BILSTM(input_dim=3*args.num_filter+args.STAT_NET_geo_embedding+len( args.STAT_NET_input_road)*args.STAT_NET_road_embedding,args=args).to(args.device).apply(weight_init)

if args.weightloss:
    weight = torch.tensor([1.0048, 1.0022, 2.9012, 1.0453, 0.7708, 0.8641, 0.7827, 1.0274]).to(args.device)
    criterion  = nn.CrossEntropyLoss(ignore_index=args.pad_value,weight=weight)
else:
    criterion  = nn.CrossEntropyLoss(ignore_index=args.pad_value)

optimizer = torch.optim.Adam([
                {'params': model_s.parameters()},
                {'params': model_g.parameters()},
                {'params': model_stat.parameters()},
                {'params': model.parameters()},
            ], lr=0.0001,weight_decay = 1e-4)
#https://blog.csdn.net/qq_43428929/article/details/126179489

# %%


# %%
best_acc = 0
for epoch in range(0, args.epochs + 1):
    t1 = time.perf_counter()
    train_loss,train_acc = train_epoch(model_s,model_g,model_stat,model,  optimizer,  criterion,train_loader)


    val_loss,val_acc  = evaluate(model_s,model_g,model_stat,model, criterion, val_loader)
    t2 = time.perf_counter()

    print('epoch:',epoch,'train loss',train_loss,'train acc',train_acc.item())
    print('epoch:',epoch,'val loss',val_loss,'val acc',val_acc.item())
    print('time {}s'.format(t2-t1))
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_s': model_s.state_dict(),
            'model_g': model_g.state_dict(),
            'model_stat': model_stat.state_dict(),
            'model': model.state_dict()
        }, args.model_save)
        print('saved best\n!!\n')

# %% [markdown]
# # 在验证集上进行最后的结果

# %%
#加载模型
#args.model_save = '/DATA1/EvolveGCN/limengyuan/datas/models/models2023-06-16 09:35:17:728874==.pth'
checkpoint = torch.load( args.model_save)
model_s.load_state_dict(checkpoint['model_s'])
model_g.load_state_dict(checkpoint['model_g'])
model_stat.load_state_dict(checkpoint['model_stat'])
model.load_state_dict(checkpoint['model'])
print(args.model_save)
# %%
for loc in  ['Hand', 'Bag', 'Hips', 'Torso']:
    print('valid',loc)
    def val_dataset(loc='Hand'):


        train_hand = cPickle.load(open('/root/autodl-tmp/valid/{}/raw_data.pkl'.format(loc),'rb'))
        stat_data = cPickle.load(open('/root/autodl-tmp/valid/{}/data.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')   

        data_x = np.arange(len(train_hand[0]))#trajectory id
        data_y = [int(i[0][0,-4]) for i in train_hand[1]]
        val_sensor = [j for i in data_x for j in train_hand[0][i]]
        val_gps = [j for i in data_x for j in train_hand[1][i]]
        val_data = dset(val_sensor,val_gps,stat_data)


        test_loader1 = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )
        '''
        train_hand = cPickle.load(open('/root/autodl-tmp/test/raw_data.pkl','rb').format(loc))
        stat_data = cPickle.load(open('/root/autodl-tmp/test/data.pkl','rb').format(loc)).fillna(0).drop_duplicates(keep='first')   

        data_x = np.arange(len(train_hand[0]))#trajectory id
        data_y = [int(i[0][0,-4]) for i in train_hand[1]]
        val_sensor = [j for i in data_x for j in train_hand[0][i]]
        val_gps = [j for i in data_x for j in train_hand[1][i]]
        val_data = dset(val_sensor,val_gps,stat_data)


        test_loader2 = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers = args.num_workers
            )
        '''
        return test_loader1
    if args.mix:
        def val_dataset(loc='Hand'):


            train_hand = cPickle.load(open('/root/autodl-tmp/valid/{}/raw_data_m.pkl'.format(loc),'rb'))
            stat_data = cPickle.load(open('/root/autodl-tmp/valid/{}/data_m.pkl'.format(loc),'rb')).fillna(0).drop_duplicates(keep='first')   

            data_x = np.arange(len(train_hand[0]))#trajectory id
            data_y = [int(i[0][0,-4]) for i in train_hand[1]]
            val_sensor = [j for i in data_x for j in train_hand[0][i]]
            val_gps = [j for i in data_x for j in train_hand[1][i]]
            val_data = dset(val_sensor,val_gps,stat_data)


            test_loader1 = torch.utils.data.DataLoader(
                dataset=val_data,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
                num_workers = args.num_workers
                )
            '''
            train_hand = cPickle.load(open('/root/autodl-tmp/test/raw_data.pkl','rb').format(loc))
            stat_data = cPickle.load(open('/root/autodl-tmp/test/data.pkl','rb').format(loc)).fillna(0).drop_duplicates(keep='first')   

            data_x = np.arange(len(train_hand[0]))#trajectory id
            data_y = [int(i[0][0,-4]) for i in train_hand[1]]
            val_sensor = [j for i in data_x for j in train_hand[0][i]]
            val_gps = [j for i in data_x for j in train_hand[1][i]]
            val_data = dset(val_sensor,val_gps,stat_data)


            test_loader2 = torch.utils.data.DataLoader(
                dataset=val_data,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
                num_workers = args.num_workers
                )
            '''
            return test_loader1
    val_loader = val_dataset(loc=loc)


    test_loss,test_acc  = evaluate(model_s,model_g,model_stat,model, criterion, val_loader)
    print('epoch:',epoch,'test loss',test_loss,'test acc',test_acc.item())


    def prediction_val(model_s,model_g,model_stat,model,loss_fn,train_dataloader):
        model_s.eval()
        model_g.eval()
        model_stat.eval()
        model.eval()

        
        losses = 0
        correct = 0
        preds = []
        idxs = []
        labels = []
        outs= []
        trip_idxs = []
        with torch.no_grad():
            for data_sensor,data_gps,stat_sensors,stat_gps,stat_road, label,idx,trip_idx, lengths in train_dataloader:
                data_sensor = data_sensor.to(args.device)
                data_gps = data_gps.to(args.device)
                stat_sensors,stat_gps,stat_road = stat_sensors.to(args.device),stat_gps.to(args.device),stat_road.to(args.device)
                label = label.to(args.device)[:,(args.L1-1):]

                output_s = model_s(data_sensor)
                output_g = model_g(data_gps)
                output_stat = model_stat(stat_sensors,stat_gps,stat_road)
                out = model(torch.cat([output_s,output_g,output_stat],dim=-1),lengths)

                loss = loss_fn( out.reshape(-1, out.shape[-1]), label.reshape(-1))
            
                losses += loss.item() 
                
                mask = label ==args.pad_value
                pred = torch.argmax(out, dim=2)
                correct +=  (pred ==label ).masked_fill(mask,0).sum().item() / (~mask).sum()

                preds.append(pred)
                idxs.append(idx[:,(args.L1-1):])
                labels.append(label)
                outs.append(out)
                trip_idxs.append(trip_idx)
                
        model_s.train()
        model_g.train()
        model.train()
        model_stat.train()
        return torch.cat(preds),torch.cat(idxs),torch.cat(trip_idxs),torch.cat(labels),torch.cat(outs),losses / len(train_dataloader),correct/len(train_dataloader)


    preds,idxs,trip_idxs,labels,outs,_,_ = prediction_val(model_s,model_g,model_stat,model, criterion, val_loader)


    idxs_1 = idxs.reshape(-1,1).cpu().numpy()
    labels_1 = labels.reshape(-1,1).cpu().numpy()
    preds_1 = preds.reshape(-1,1).cpu().numpy()
    outs_1 = outs.reshape(-1,outs.shape[2]).cpu().numpy()

    results = pd.DataFrame(np.concatenate((idxs_1,labels_1,preds_1,outs_1), axis =1))
    results.columns = ['idx','label_true','preds'] + [i for i in range(8)]


    #results_new = results.groupby('id')[[i for i in range(8)]].mean()
    results_new = results.groupby('idx')[['label_true']].min().reset_index().astype(int)
    results_new['out'] = results.groupby('idx')[[i for i in range(8)]].mean().idxmax(axis=1).values
    results_new['preds'] = results.groupby('idx')[['preds']].agg(lambda x: x.value_counts().index[0]).values.astype(int)



    results_new = results_new[results_new['idx']>=0]
    print('acc_pred',(results_new['label_true']==results_new['preds']).mean())
    print('acc_outs',(results_new['label_true']==results_new['out']).mean())


    y_true = results_new['label_true']
    y_pred = results_new['preds']
    label_sort = ['Still','Walk','Run','Bike','Car','Bus','Train', 'Subway']
    print(classification_report(y_true, y_pred, target_names = label_sort))

    # %%
    import matplotlib.pyplot as plt
    import numpy as np

    # 示例混淆矩阵
    confusion_matrix_ = confusion_matrix(y_true, y_pred)

    # 类别标签
    labels = label_sort

    # 绘制混淆矩阵
    plt.imshow(confusion_matrix_, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # 设置刻度标签
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = confusion_matrix_.max() / 2
    for i in range(confusion_matrix_.shape[0]):
        for j in range(confusion_matrix_.shape[1]):
            plt.text(j, i, format(confusion_matrix_[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix_[i, j] > thresh else "black")

    # 显示图形
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


