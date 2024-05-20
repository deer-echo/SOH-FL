# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from utils.dataset_utils import check, separate_data, split_data, save_file
from torch.utils.data import Dataset, DataLoader

random.seed(1)
np.random.seed(1)
num_clients = 9
num_classes = 9
dir_path = "CICIDS2017QuantityFeature3/"


#generate reduced data
def generate_reduced():
    # Get CIC data
    # 从CSV文件中读取数据
    data = pd.read_csv('CICIDS2017/CICIDS2017_pre_all_noInf.csv')

    #################对数据集进行随机抽样，保留30%的数据######################
    # sampled_data = data.sample(frac=0.3, random_state=1) #抽样成80万条
    # data=sampled_data

    ##################去除总数<1000的label####################
    # 首先，计算每个类别的总数
    class_counts = data['Activity'].value_counts()
    print("number of every class:", class_counts)
    # 然后，找出总数大于等于1000的类别
    valid_classes = class_counts[class_counts >= 5000].index
    # 最后，根据筛选出的类别过滤数据框
    filtered_df = data[data['Activity'].isin(valid_classes)]
    # 首先，获取不同的Activity值
    unique = filtered_df['Activity'].unique()
    # 创建一个映射字典，将每个Activity映射成一个数字
    activity_mapping = {activity: i for i, activity in enumerate(unique)}
    # 使用映射字典将Activity列的值标记成数字
    filtered_df['Activity'] = filtered_df['Activity'].map(activity_mapping)
    ##########################创建label列用于二分#########################
    df = pd.DataFrame(filtered_df)
    # df['label'] = df['Activity'].apply(lambda x: 0 if x == 0 else 1)

    #########################对每个activity抽样1000条，保证quantity一致##############################
    # df = df.groupby('Activity').apply(lambda x: x.sample(n=5000, replace=True))

    data = df.to_numpy()
    # 去掉数量小于500的label
    np.savetxt('CICIDS2017/reduced_CIC_data.csv', data, delimiter=',', fmt='%d')
# Allocate data to users
def generate_CIC(dir_path, num_clients, num_classes, niid, balance, partition,zero_day=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    test_Spath = dir_path + "Stest/"

    if check(config_path, train_path, test_Spath,test_path, num_clients, num_classes, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    if not os.path.exists('CICIDS2017/reduced_CIC_data.csv'):
        generate_reduced()
    data = pd.read_csv('CICIDS2017/reduced_CIC_data.csv')
    data = data.to_numpy()


    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # data=torch.tensor(data).reshape(1,-1) #将向量变形为行向量
    # normalized_data = 2*(data - data.min()) / (data.max() - data.min()) - 1

    X, y, statistic = separate_data((data[:,:-1],data[:,-1]), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=9)
    # train test data分割
    train_data, test_Sdata, test_data = split_data(X, y,zero_day)

    save_file(config_path, train_path, test_Spath, test_path, train_data, test_Sdata, test_data, num_clients,
              num_classes,
              statistic, niid, balance, partition,zero_day)

if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_CIC(dir_path, num_clients, num_classes, niid=False, balance=False,partition='pat',zero_day=True)