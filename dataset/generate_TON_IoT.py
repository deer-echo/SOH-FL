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
from sklearn.preprocessing import LabelEncoder


df=pd.DataFrame


random.seed(1)
np.random.seed(1)
num_clients = 12
num_classes = 9
dir_path = "TON-IoT/"


#按原始的http分割数据
def seperate_by_id(data):
    # 创建三个空表
    X=[]
    y=[]
    grouped_data = data.groupby('src_ip')
    for group in grouped_data:
        dfx=group[1].drop('type',axis=1)
        dfx=dfx.drop('src_ip',axis=1)
        dfy=group[1]['type']
        X.append(dfx.values)
        y.append(dfy.values)

    return X, y


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
    data = pd.read_csv('TON-IoT/TON-IoT.csv')
    data = pd.DataFrame(data)


    X, y = seperate_by_id(data)

    # train test data分割
    train_data, test_Sdata, test_data = split_data(X, y,zero_day)
    statistic=[]
    save_file(config_path, train_path, test_Spath, test_path, train_data, test_Sdata, test_data, num_clients,
              num_classes,
              statistic, niid, balance, partition,zero_day)

if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None
    generate_CIC(dir_path, num_clients, num_classes, niid=False, balance=False,partition='pat',zero_day=False)