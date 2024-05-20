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

import copy
import time
import numpy as np
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from threading import Thread


class PerAvg(Server):
    #初始化
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPerAvg)
        #输出用户比例、用户总数量
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):#全局轮次
            s_t = time.time() #记录当前时间
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                print('selected client:',client.id)
            #########################server向client发送模型############################send all parameter for clients
            self.send_models()
            #几轮评估一次鸭
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model with one step update")
                self.evaluate_one_step()
            #对选定客户端进行训练
            # choose several clients to send back upated model to server
            #为什么两个client.train()
            for client in self.selected_clients:
                client.train()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            ######################接收来自客户端的更新后的模型参数###########################
            self.receive_models()
            #如果需要进行动态全局模型评估，执行评估。
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            #聚合客户端的参数
            self.aggregate_parameters()
            #记录训练时间成本
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1]) #最后一个的时间
            #如果满足一定条件，提前结束训练循环。
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        #输出最佳准确率和每轮平均时间成本。
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))


        self.save_global_model()
        self.save_all_local()
        self.save_results()


        #如果有新客户端加入，进行额外的评估和微调
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPerAvg)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    #评估方法
    def evaluate_one_step(self, acc=None, loss=None):
        models_temp = []
        test_metrix_everyone=[]
        for c in self.clients: #遍历各个用户
            #对每个客户端的模型进行深拷贝，并将副本存储在 models_temp 列表中。
            models_temp.append(copy.deepcopy(c.model))
            #对每个客户端执行一次训练步骤。
            c.train_one_step()
        #获取测试集上的性能指标。
        stats = self.test_metrics()
        # set the local model back on clients for training process
        #使用 enumerate 函数遍历 self.clients 中的客户端对象，
        # 同时获取索引 i 和客户端对象 c
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)
        #调用 train_metrics 方法计算训练集的性能指标，并将结果存储在 stats_train 中
        stats_train = self.train_metrics()
        # set the local model back on clients for training process
        for i, c in enumerate(self.clients):
            #调用客户端对象 c 的 clone_model 方法，
            # 将 models_temp[i] 中的临时模型复制到客户端的模型中。
            c.clone_model(models_temp[i], c.model)
        #计算每个客户端的准确率，并将结果存储在 accs 中
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        print('accs:',accs)
        #计算所有客户端的平均测试准确率，并将结果存储在 test_acc 中
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_precision =sum(stats[4])/sum(stats[1])
        test_recall = sum(stats[5]) / sum(stats[1])
        test_f1 = sum(stats[6]) / sum(stats[1])
        test_metrix=sum(stats[7]) #直接相加即可
        test_metrix_everyone=test_metrix_everyone.append(stats[7])
        #计算所有客户端的平均训练损失，并将结果存储在 train_loss 中
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])


        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            print("accuracy none")
            acc.append(test_acc)
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            print("loss none")
            loss.append(train_loss)

        self.rs_test_precision.append(test_precision)
        self.rs_test_recall.append(test_recall)
        self.rs_test_f1.append(test_f1)
        self.rs_test_metrix.append(test_metrix)
        self.rs_test_metrix_everyone.append(test_metrix_everyone)


        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test Precision: ",test_precision)
        print("Averaged Test recall: ",test_recall)
        print("Averaged Test f1: ",test_f1)
        print("Averaged Test confusion matrix: \n",test_metrix)
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("confusion matrix for all clients")