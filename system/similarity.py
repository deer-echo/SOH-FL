import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch.optim as optim
import time
##################通用####################
round=1#检验次数
input_size = 31
# 定义损失函数和优化器
criterion = nn.MSELoss()
# criterion = nn.BCELoss()
###############初始化##################
correct = [5, 6, 7, 4, 3, 0, 8, 1, 2,11,10,9]
# correct = [5, 6, 7, 4, 3, 0, 8, 1, 2]
client_number=12
really_correct_time=0
three_correct_time = 0

###########################data preparing#####################
# 创建一个空列表，用于存储结构化数据
train_listX=[]
train_listY=[]
support_listX=[]
support_listY=[]
#缩放数据
scaler=MinMaxScaler()
# filename="CICIDS2017QuantityFeature3"
# filename='IOT-23'
filename="TON-IoT"
# 读取0.npz到8.npz的train文件 or 0-11
for i in range(client_number):
    xfile_name = f"../dataforsimi/{filename}/train/{i}x.csv"
    x = np.loadtxt(xfile_name,delimiter=',')

    yfile_name = f"../dataforsimi/{filename}/train/{i}y.csv"
    y = np.loadtxt(yfile_name, delimiter=',')
    # 随机抽取1000行数据
    indices = np.random.choice(x.shape[0],1000,replace=False)
    # 从 x 和 y 中各取出1000行数据
    sampled_x = x[indices]
    sampled_y = y[indices]
    x_scaled = scaler.fit_transform(sampled_x)
    train_listX.append(x_scaled)
    train_listY.append(sampled_y)
# 读取0.npz到8.npz的support文件
for i in range(client_number):
    xfile_name = f"../dataforsimi/{filename}/Stest/{i}x.csv"
    x = np.loadtxt(xfile_name,delimiter=',')
    x_scaled = scaler.fit_transform(x)
    yfile_name = f"../dataforsimi/{filename}/Stest/{i}y.csv"
    y = np.loadtxt(yfile_name, delimiter=',')
    support_listX.append(x_scaled)
    support_listY.append(y)

###################################auto############################
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_target, sparsity_weight):
        super(SparseAutoencoder, self).__init__()
        # self.encoder = nn.Linear(input_size, hidden_size)
        # self.decoder = nn.Linear(hidden_size, input_size)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.encoder = nn.Sequential(
                    nn.Linear(input_size, 128),
                    # nn.Dropout(p=0.2),#可以更换dropout所在的层数 0.2-0.5
                    nn.ReLU(True),
                    nn.Linear(128, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 12),
                    nn.ReLU(True),
                    nn.Linear(12, hidden_size))
        self.decoder = nn.Sequential(
                    nn.Linear(hidden_size, 12),
                    nn.ReLU(True),
                    nn.Linear(12, 64),
                    nn.ReLU(True),
                    nn.Linear(64, 128),
                    nn.ReLU(True),
                    nn.Linear(128, input_size),
                    nn.Sigmoid())
        # self.dropout = nn.Dropout(p=0.1) #丢弃20%的神经元 通常在0.2-0.5之间
    def forward(self, x):
        # 编码
        hidden = self.encoder(x)
        #dropout
        # hidden = self.dropout(hidden)
        # 解码
        output = self.decoder(hidden)
        return output, hidden


# 定义KL散度函数
def kl_divergence(p, q):
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

#loss = reconstruction
# 设置参数
##########AE参数###################
num_epochs = 100 #每轮的epoch数
batch_size = 128
learning_rate = 0.001 #优化器的学习率 0.001左右
hidden_size = 10 #压缩后的层数 3~30
##################sparse相关################
sparsity_target = 0.1 #KL散度的稀疏度
sparsity_weight = 0.1 #KL散度的权重
#################cosine相关##############
cosine_weight = 0.1



true_label = []
predict_label = []

start_time = time.time()
######################################set data#################################
for j in range(0,round):
    print("=============round ",j,"=============")
    ############################## 初始化稀疏自编码器######################################
    autoencoder = SparseAutoencoder(input_size, hidden_size, sparsity_target, sparsity_weight)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    tensor_list=[]
    for arr in train_listX:
        tensor=torch.tensor(arr,dtype=torch.float32)
        tensor_list.append(tensor)
    sample_tensor=[]
    for arr in support_listX:
        tensor=torch.tensor(arr,dtype=torch.float32)
        sample_tensor.append(tensor)

    encoded_data=[]
    num=0
    # 训练模型
    for dataset in tensor_list:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for data in dataloader:
                inputs = data
                optimizer.zero_grad()
                outputs, hidden = autoencoder(inputs)
                ##################纳入cosin作为loss的一部分######################
                # 获得对应子集及其tensor
                sub = list(train_listX[num])
                sub_data = random.sample(sub, 128)
                sub_tensor = torch.tensor(sub_data, dtype=torch.float32)
                # 计算降维后的数据，及其cosine，作为loss的一部分
                mid_encode = autoencoder.encoder(dataset).detach().numpy()
                mid_sub_encode = autoencoder.encoder(sub_tensor).detach().numpy()
                # cos=np.dot(mid_encode, mid_sub_encode) / (np.linalg.norm(mid_encode) * np.linalg.norm(mid_sub_encode))
                cosin_simi = np.array(
                    [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in
                     zip(mid_encode, mid_sub_encode)])
                average_cosin = np.mean(cosin_simi)
                #############################################################
                # 重构损失
                reconstruction_loss = criterion(outputs, inputs)
                # 计算编码层的平均激活值
                avg_activation = torch.mean(hidden, dim=1)
                # 计算KL散度
                sparsity_loss = torch.sum(kl_divergence(sparsity_target, avg_activation))
                # print('===============KL loss: ',sparsity_loss*sparsity_weight)
                # 总损失 = 重构损失+KL散度+cosine相似度损失
                loss = (reconstruction_loss
                        +cosine_weight*(1-average_cosin))
                        # +sparsity_loss*sparsity_weight)
                # loss = reconstruction_loss + sparsity_weight * sparsity_loss
                # loss = reconstruction_loss
                loss.backward()
                optimizer.step()

            print(f'datasets Epoch {epoch + 1}, Loss: {loss.item()}',f'round: {j}')

        encode = autoencoder.encoder(dataset).detach().numpy()
        encoded_data.append(encode)
        num += 1
    num=0
    encode_sample=[]
    for dataset2 in sample_tensor:
        dataloader = DataLoader(dataset2, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            for data in dataloader:
                # 获得对应子集及其tensor
                sub = list(support_listX[num])
                sub_data = random.sample(sub, 128)
                sub_tensor = torch.tensor(sub_data, dtype=torch.float32)
                # 计算降维后的数据，及其cosine，作为loss的一部分
                mid_encode = autoencoder.encoder(dataset).detach().numpy()
                mid_sub_encode = autoencoder.encoder(sub_tensor).detach().numpy()
                # cos=np.dot(mid_encode, mid_sub_encode) / (np.linalg.norm(mid_encode) * np.linalg.norm(mid_sub_encode))
                cosin_simi = np.array(
                    [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in
                     zip(mid_encode, mid_sub_encode)])
                average_cosin = np.mean(cosin_simi)
                ##########################cosin################
                inputs = data
                optimizer.zero_grad()
                outputs, hidden = autoencoder(inputs)
                # 重构损失
                reconstruction_loss = criterion(outputs, inputs)
                # 计算编码层的平均激活值
                avg_activation = torch.mean(hidden, dim=1)
                # 计算KL散度
                sparsity_loss = torch.sum(kl_divergence(sparsity_target, avg_activation))
                # print('===============KL loss: ',sparsity_loss*sparsity_weight)
                # 总损失
                loss = (reconstruction_loss
                        +cosine_weight*(1-average_cosin))
                        #+sparsity_loss*sparsity_weight
                # loss=reconstruction_loss
                loss.backward()
                optimizer.step()
            print(f'sample Epoch {epoch + 1}, Loss: {loss.item()}',f'round: {j}')
        encode2 = autoencoder.encoder(dataset2).detach().numpy()
        encode_sample.append(encode2)

    #####################cosin similarity###########################
    count = 0
    maxLabel=[]
    for q in encode_sample:
        result = []
        for v in encoded_data:
            cosin_simi = np.array(
                [np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(q, v)])
            average_cosin = np.mean(cosin_simi)
            result.append(average_cosin)
        print(result)
        # 找到最大值的索引
        max_index = result.index(max(result))
        predict_label.append(max_index)
        true_label.append(correct[count])
        # 打印最大值的下标
        print("predict: ", max_index)
        if max_index==correct[count]:
            really_correct_time+=1
        print('actually', correct[count])
        #获得最大的三个值的下标

        three_values = sorted(((value, index) for index, value in enumerate(result)), reverse=True)[:3]
        # 输出结果
        for value, index in three_values:
            print(f"值: {value}, 索引: {index}")
            if index==correct[count]:
                three_correct_time+=1

        # maxLabel.append(three_values)
        print('one correct time: ', really_correct_time)
        # print('two correct time: ', two_correct_time)
        print('three correct time: ', three_correct_time)
        # print('four correct time: ', four_correct_time)
        # print('five correct time: ', five_correct_time)
        # print('six correct time: ', six_correct_time)
        # print('seven correct time: ', seven_correct_time)
        # print('eight correct time: ', eight_correct_time)




        count += 1

end_time=time.time()
print("runtime: ",end_time-start_time)

##############混同矩阵##################
# cm = confusion_matrix(true_label, predict_label)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
#             xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7',
#                          'Class 8'],
#             yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7',
#                          'Class 8'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
# print(cm)


