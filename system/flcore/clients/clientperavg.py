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
import torch
import time
import copy
from flcore.optimizers.fedoptimizer import PerAvgOptimizer
from flcore.clients.clientbase import Client


class clientPerAvg(Client):
    #初始化
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # self.beta = args.beta
        self.beta = self.learning_rate
        #优化器
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        #学习率调度器
        #这个类似乎是用于实现特定的客户端行为，
        # 包括学习率的设置、优化器的初始化和学习率的调度
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,  #设置优化器
            gamma=args.learning_rate_decay_gamma #设置学习率衰减
        )

    def train(self):
        #加载训练数据
        trainloader = self.load_train_data(self.batch_size*2)
        #设置初始时间
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        #确定最大本地轮次
        max_local_epochs = self.local_epochs
        #慢速模式
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):  # local update
            for X, Y in trainloader:
                #首先复制当前模型的参数作为临时模型参数备份。
                temp_model = copy.deepcopy(list(self.model.parameters()))

                # step 1
                #针对第一个批次数据进行前向传播、计算损失、反向传播和参数更新。
                if type(X) == type([]): #是列表
                    x = [None, None] #创建一个列表 x，包含两个元素，初始化为 None。
                    #(support set)将输入数据的两部分分别赋值给 x 的两个元素，并将第一个部分转移到指定的设备（如GPU）上
                    x[0] = X[0][:self.batch_size].to(self.device)
                    x[1] = X[1][:self.batch_size]
                else: #如果 X 的类型不是列表，表示输入数据只有一部分。在这种情况下：
                    #将输入数据的前 batch_size 个样本转移到指定的设备上。
                    x = X[:self.batch_size].to(self.device)
                #将标签数据 Y 的前 batch_size 个样本转移到指定的设备上
                y = Y[:self.batch_size].to(self.device)
                #慢速训练则加入延时
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))


                #计算模型输出与标签之间的损失。
                output = self.model(x)
                loss = self.loss(output, y)
                #对模型参数的梯度进行清零。
                self.optimizer.zero_grad()
                #执行反向传播计算梯度。
                loss.backward()
                #根据梯度更新模型参数。第一轮β为0，使用alpha进行内部更新
                self.optimizer.step()
                #print("after optimizer1 client model:",self.model.fc.bias)


                # step 2
                #第一轮和第二轮的区别：
                #1. 学习率，一个是内部更新的学习率α，另一个是外部更新的学习率β
                #2. 使用的数据不同，一个是support set，一个是query set
                #3. 基于的w不一样，第二个w是一的结果
                #针对第二个批次数据进行前向传播、计算损失、反向传播，
                # 但在更新参数之前，将模型参数恢复到第一步更新之前的状态。
                if type(X) == type([]):
                    x = [None, None]
                    x[0] = X[0][self.batch_size:].to(self.device)
                    x[1] = X[1][self.batch_size:]
                else:
                    x = X[self.batch_size:].to(self.device)
                y = Y[self.batch_size:].to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # restore the model parameters to the one before first update
                for old_param, new_param in zip(self.model.parameters(), temp_model):
                    old_param.data = new_param.data.clone()
                #第二轮使用beta进行外部更新
                self.optimizer.step(beta=self.beta)
                #print("after optimizer2 client model:",self.model.fc.bias)

        # self.model.cpu()
        #学习率衰减
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_one_step(self):
        #加载测试数据集，并按照指定的批量大小进行分批处理。
        testSloader = self.load_test_Sdata(self.batch_size)
        #加载训练数据集，并按照指定的批量大小进行分批处理。
        iter_loader = iter(testSloader)
        # self.model.to(self.device)
        #将模型设置为训练模式，这会启用模型中的训练相关功能，如Dropout和Batch Normalization。
        self.model.train()
        #从数据集迭代器中获取下一个训练数据批次。
        (x, y) = next(iter_loader)
        #如果输入数据 x 是一个列表，则将列表中的第一个元素移动到指定的设备（如GPU）上
        if type(x) == type([]):
            #否则，将输入数据 x 移动到指定的设备上
            x[0] = x[0].to(self.device)
        else:
            #将标签数据 y 移动到指定的设备上。
            x = x.to(self.device)
        y = y.to(self.device)
        #将输入数据 x 输入到模型中进行前向传播，得到模型的输出结果。
        output = self.model(x)
        #计算模型输出与标签数据之间的损失。
        loss = self.loss(output, y)
        #print('loss',loss)
        #清除之前的梯度，以便进行新一轮的反向传播
        self.optimizer.zero_grad()
        #执行反向传播，计算模型参数的梯度
        loss.backward()
        #根据计算得到的梯度，更新模型参数。
        self.optimizer.step()

        # self.model.cpu()

    # def train_on_test(self):
    #     # 加载训练数据集，并按照指定的批量大小进行分批处理。
    #     trainloader = self.load_train_data(self.batch_size)
    #     # 加载训练数据集，并按照指定的批量大小进行分批处理。
    #     iter_loader = iter(trainloader)
    #     # self.model.to(self.device)
    #     # 将模型设置为训练模式，这会启用模型中的训练相关功能，如Dropout和Batch Normalization。
    #     self.model.train()
    #     # 从数据集迭代器中获取下一个训练数据批次。
    #     (x, y) = next(iter_loader)
    #     # 如果输入数据 x 是一个列表，则将列表中的第一个元素移动到指定的设备（如GPU）上
    #     if type(x) == type([]):
    #         # 否则，将输入数据 x 移动到指定的设备上
    #         x[0] = x[0].to(self.device)
    #     else:
    #         # 将标签数据 y 移动到指定的设备上。
    #         x = x.to(self.device)
    #     y = y.to(self.device)
    #     # 将输入数据 x 输入到模型中进行前向传播，得到模型的输出结果。
    #     output = self.model(x)
    #     # 计算模型输出与标签数据之间的损失。
    #     loss = self.loss(output, y)
    #     # 清除之前的梯度，以便进行新一轮的反向传播
    #     self.optimizer.zero_grad()
    #     # 执行反向传播，计算模型参数的梯度
    #     loss.backward()
    #     # 根据计算得到的梯度，更新模型参数。
    #     self.optimizer.step()
    #
    #     # self.model.cpu()

    def train_metrics(self, model=None):
        trainloader = self.load_train_data(self.batch_size*2)
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        for X, Y in trainloader:
            # step 1
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][:self.batch_size].to(self.device)
                x[1] = X[1][:self.batch_size]
            else:
                x = X[:self.batch_size].to(self.device)
            y = Y[:self.batch_size].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

            # step 2
            if type(X) == type([]):
                x = [None, None]
                x[0] = X[0][self.batch_size:].to(self.device)
                x[1] = X[1][self.batch_size:]
            else:
                x = X[self.batch_size:].to(self.device)
            y = Y[self.batch_size:].to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            self.optimizer.zero_grad()
            output = self.model(x)
            loss1 = self.loss(output, y)

            train_num += y.shape[0]
            losses += loss1.item() * y.shape[0]

        return losses, train_num

    def train_one_epoch(self):
        trainloader = self.load_train_data(self.batch_size)
        for i, (x, y) in enumerate(trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            output = self.model(x)
            loss = self.loss(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()