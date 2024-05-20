# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import math
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
# 实现异构的包
import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import random

batch_size = 400  # 用于support set的数量
train_size = 0.7  # merge original training set and test set, then split it manually.
least_samples = 1  # guarantee that each client must have at least one samples for testing.
alpha = 0.1  # for Dirichlet distribution


def check(config_path, train_path, test_Spath, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None):
    # check existing dataset
    # 这部分代码首先检查指定的config_path是否存在。
    # 如果存在，它尝试打开文件并将其内容加载到config变量中。
    # 然后，它检查config中的值是否与传入的参数匹配。如果匹配，则打印消息并返回True。
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True
    # 这部分代码用于检查训练集和测试集的目录是否存在，
    # 如果不存在，则创建这些目录。
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_Spath)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 如果之前的条件都没有满足，函数将返回False
    return False


# 控制异构
def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    # 创建三个空表
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]
    # 将传入的data解包为dataset_content和dataset_label，
    # 假设data是一个包含数据和标签的元组
    dataset_content, dataset_label = data
    # 创建一个空的字典dataidx_map，用于存储数据索引的映射关系。
    dataidx_map = {}

    if not niid:  # 若为iid
        partition = 'pat'  # 设置为'pat'
        class_per_client = num_classes  # 每个用户中的类别均为所有类别

    if partition == 'pat':  # 若为pathological
        ###########获得每个class的索引列表###################
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            # 数据按类别分配给客户端
            idx_for_each_class.append(idxs[dataset_label == i])
        # 创建了一个名为class_num_per_client的列表，它包含了num_clients个元素，·
        # 每个元素的值都是class_per_client，用于表示每个客户端需要的类别数量。
        class_num_per_client = [class_per_client for _ in range(num_clients)]
        # 根据每个客户端需要的类别数量，选择合适的客户端来参与数据划分。
        for i in range(num_classes):  # 首先遍历每个类别
            selected_clients = []
            for client in range(num_clients):  # 根据每个客户端需要的类别数量
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)  # 然后根据每个客户端需要的类别数量，选取合适的客户端参与数据划分
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]
            num_all_samples = len(idx_for_each_class[i])  # 每个类别的样本数
            num_selected_clients = len(selected_clients)  # 被选择的客户端数量
            num_per = num_all_samples / num_selected_clients  # 每个客户端应得的样本数
            if balance:  # 如果为True
                # 则将num_per转换为整数，并复制num_selected_clients-1次
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:  # 为False
                # 则生成客户端数量-1个，大小介于 每个用户样本数/类别大小和最小样本数/类别大小 的大者和 每个用户应得到的样本之间（这个不太科学）
                num_samples = np.random.randint(max(num_per / num_classes, least_samples / num_classes), num_per + 1,
                                                num_selected_clients - 1).tolist()
            # 将剩余的样本数加入num_samples列表中。
            num_samples.append(num_all_samples - sum(num_samples))
            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                # 迭代selected_clients和num_samples来分配样本给每个客户端，
                if client not in dataidx_map.keys():  # 检查client是否已经在dataidx_map中
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]  # 在则分配数据
                else:  # 否则
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # 使用狄利克雷分布进行数据划分
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    # 对每个客户端进行数据分配，并统计每个客户端的数据情况，包括数据内容和标签。
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    # 删除data变量，释放内存。
    del data
    # gc.collect()

    for client in range(num_clients):
        # 打印每个客户端的数据情况
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    # 返回划分后的数据集X、标签y和统计信息statistic。
    return X, y, statistic


# 将数据按train_size比例分割成train和test数据，默认为0.7
def split_data(X, y, zero_day=False):
    # Split dataset
    # 创建了两个空列表train_data和test_data，
    # 以及一个字典num_samples，用于存储训练集和测试集的数据，以及每个数据集的样本数量。
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}
    # 遍历标签y的长度，对每个客户端的数据进行划分
    for i in range(len(y)):
        # 在这里，使用train_test_split函数将每个客户端的特征和标签划分为训练集和测试集，
        # 并将划分后的数据分别赋值给X_train、X_test、y_train和y_test。
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)
        # 在这里，将划分后的训练集和测试集数据以字典形式加入train_data和test_data列表中，
        # 并记录每个数据集的样本数量。
        ##############实现 zero_day#################
        if zero_day:
            y_train = y_train.astype(int)
            keep = np.where(y_train != i)
            y_train = y_train[keep]
            X_train = X_train[keep]
            keep = np.where(y_train != (i + 1) % 9)
            y_train = y_train[keep]
            X_train = X_train[keep]
            keep = np.where(y_train != (i + 2) % 9)
            y_train = y_train[keep]
            X_train = X_train[keep]
            # keep = np.where(y_train != (i + 3) % 9)
            # y_train = y_train[keep]
            # X_train = X_train[keep]
            # keep = np.where(y_train != (i + 4) % 9)
            # y_train = y_train[keep]
            # X_train = X_train[keep]
            # keep = np.where(y_train != (i + 5) % 9)
            # y_train = y_train[keep]
            # X_train = X_train[keep]
            y_test = y_test.astype(int)

            keep = np.where(y_test != i)
            y_test = y_test[keep]
            X_test = X_test[keep]
            y_test = y_test.astype(int)
            keep = np.where(y_test != (i+1)%9)
            y_test = y_test[keep]
            X_test = X_test[keep]
            keep = np.where(y_test != (i+2)%9)
            y_test = y_test[keep]
            X_test = X_test[keep]
            # keep = np.where(y_test != (i+3)%9)
            # y_test = y_test[keep]
            # X_test = X_test[keep]
            # keep = np.where(y_test != (i+4)%9)
            # y_test = y_test[keep]
            # X_test = X_test[keep]
            # keep = np.where(y_test != (i+5)%9)
            # y_test = y_test[keep]
            # X_test = X_test[keep]
            y_test = y_test.astype(int)
        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    ################将test_data打乱顺序###########################
    #####################使得test data和train不重合##################
    copy = test_data[:]  # 复制
    original_indices = list(range(len(test_data)))
    flag = 1
    combined = list(zip(test_data, num_samples['test']))
    # 目前固定死，为了实验方便
    original_indices=[5, 6, 7, 4, 3, 0, 8, 1, 2, 11,10,9]
    original_indices_sorted=original_indices
    combined = [combined[i] for i in original_indices]
    test_data, num_samples['test'] = zip(*combined)
    num_samples['test'] = list(num_samples['test'])
    #######
    # while flag:
    #     for item1, item2 in zip(test_data, copy):
    #         if np.array_equal(item1, item2):
    #             # print('same')
    #             random.shuffle(original_indices)
    #             # random.shuffle(combined)
    #             combined = [combined[i] for i in original_indices]
    #             test_data, num_samples['test'] = zip(*combined)
    #             num_samples['test'] = list(num_samples['test'])
    #             flag = 1
    #             # 获取重新排列后test_data的原始下标
    #             original_indices_sorted = [original_indices.index(i) for i in range(len(test_data))]
    #             break
    #         # else:
    #         #     print('not same')
    #         flag = 0
    print("+++++++++++++++++++++++++indice after sorting:",original_indices_sorted,"++++++++++++++++++++++++++++")

    ###################把test data 分成 support set和query set MNIST###################################
    # 取前batch_size个
    # test_Sdata = tuple({
    #                        'x':item['x'][:batch_size],
    #                        'y':item['y'][:batch_size]
    #                    } for item in test_data)
    ##########################################CIC#####################################
    test_Sdata = []
    rest_data = []
    # 遍历test_data中的每个字典
    for d in test_data:
        selected_x = np.empty((0, 31))  # 初始化65||25列的空array
        selected_y = np.empty((0))  # 初始化为一维空array
        rest_x = d['x']
        rest_y = d['y']
        for i in range(batch_size):
            test_data_len = len(rest_y)
            random_indice = int(random.randint(0, test_data_len - 1))
            print(i, " indice selected:", random_indice)
            selected_x = np.vstack([selected_x, d['x'][random_indice]])
            selected_y = np.append(selected_y, d['y'][random_indice])
            rest_x = np.delete(rest_x, random_indice, axis=0)
            rest_y = np.delete(rest_y, random_indice)
            # selected_y=np.concatenate(selected_y,d['y'][random_indice])
        data_dict = {'x': selected_x, 'y': selected_y}
        data_rest_dict = {'x': rest_x, 'y': rest_y}
        test_Sdata.append(data_dict)
        rest_data.append(data_rest_dict)
    test_Sdata = tuple(test_Sdata)
    # test_data = tuple(rest_data)
    ##################移除test_data中与test_Sdata重合的数据作为query set##########################

    # test_data=miss_data
    # test_data = tuple(item for item in test_data if item not in test_Sdata.all())
    # test_Sdata_x = [item['x'][:batch_size] for item in test_data]
    # test_Sdata_y = [item['y'][:batch_size] for item in test_data]
    # test_data = [item for item in test_data if item['x'] not in test_Sdata_x or item['y'] not in test_Sdata_y]
    ################ 对每种label按比例获得##############################
    #     unique_y = np.unique(d['y'])  # 获取当前字典中y值的唯一值
    #     selected_x=np.empty((0,65))#初始化65列的空array
    #     selected_y=np.empty((0))#初始化为一维空array
    #     rest_x = np.empty((0, 65))  # 初始化65列的空array
    #     rest_y = np.empty((0))  # 初始化为一维空array
    #     for y_value in unique_y:
    #         # 找到对应y值的索引
    #         indices = np.where(d['y'] == y_value)[0]
    #         total_num = len(indices)
    #         selected_num = math.ceil(batch_size*total_num)
    #         print('==================total num of',y_value,' is ',total_num)
    #         # 取出对应y值的前batch_size个x和y
    #         selected_x = np.concatenate((selected_x,d['x'][indices[:selected_num]]))
    #         selected_y = np.concatenate((selected_y,d['y'][indices[:selected_num]]))
    #         rest_x=np.concatenate((rest_x,d['x'][indices[selected_num:]]))
    #         rest_y=np.concatenate((rest_y,d['y'][indices[selected_num:]]))
    #         # 将dict放入test_Sdata
    #     data_dict = {'x': selected_x, 'y': selected_y}
    #     data_rest={'x': rest_x, 'y': rest_y}
    #     test_Sdata.append(data_dict)
    #     test_rest.append(data_rest)
    # test_Sdata = tuple(test_Sdata)
    # test_data=tuple(test_rest) 暂时让support set 呆在query
    ######################################end for CIC#########################################################

    # 从字典中提取值并创建NumPy数组
    print("train labels")
    cli_num = 0
    for d in train_data:
        print('\n---------------------client', cli_num, '------------------------')
        p = d['y']
        unique_values, counts = np.unique(p, return_counts=True)
        # 打印每个值和对应的个数
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} ||", end='')
        cli_num += 1
    print("\n test labels")
    cli_num = 0
    for d in test_data:
        print('\n---------------------client', cli_num, '------------------------')
        p = d['y']
        unique_values, counts = np.unique(p, return_counts=True)
        # 打印每个值和对应的个数
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} ||", end='')
        cli_num += 1
    print("\n support labels")
    # print(np.unique(test_Sdata['y']))
    cli_num = 0
    for d in test_Sdata:
        print('\n---------------------client', cli_num, '------------------------')
        p = d['y']
        unique_values, counts = np.unique(p, return_counts=True)
        # 打印每个值和对应的个数
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} ||", end='')
        cli_num += 1
    print("\nTotal number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    result = [x - batch_size for x in num_samples['test']]
    print("The number of test samples:", result)  # 去除support set
    del X, y
    # gc.collect()

    return train_data, test_Sdata, test_data


# 将分割后的test和train数据保存到对应路径
def save_file(config_path, train_path, test_Spath, test_path, train_data, test_Sdata, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None, zero_day=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
        'zero_day': zero_day
    }

    # gc.collect()
    print("Saving to disk.\n")

    # 这个循环遍历train_data列表中的元素，每个元素都是一个字典，
    # 然后将这个字典保存为一个压缩的.npz文件，文件名由idx决定，保存在train_path指定的路径下。
    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
        file_name = train_path + str(idx) + 'x.csv'
        np.savetxt(file_name, train_dict['x'], delimiter=',')
        file_name = train_path + str(idx) + 'y.csv'
        np.savetxt(file_name, train_dict['y'], delimiter=',')
    # 这个循环遍历test_data列表中的元素，每个元素也是一个字典，
    # 然后将这个字典保存为一个压缩的.npz文件，文件名由idx决定，保存在test_path指定的路径下。
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
        file_name = test_path + str(idx) + 'x.csv'
        np.savetxt(file_name, test_dict['x'], delimiter=',')
        file_name = test_path + str(idx) + 'y.csv'
        np.savetxt(file_name, test_dict['y'], delimiter=',')
    # 同理保存support test数据
    for idx, test_Sdict in enumerate(test_Sdata):
        with open(test_Spath + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_Sdict)
        file_name = test_Spath + str(idx) + 'x.csv'
        np.savetxt(file_name, test_Sdict['x'], delimiter=',')
        file_name = test_Spath + str(idx) + 'y.csv'
        np.savetxt(file_name, test_Sdict['y'], delimiter=',')
    # 这段代码将config字典以json格式保存到config_path指定的文件中
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
