import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('')
from system.flcore.servers.serverperavg import PerAvg
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from flcore.servers.serverperavg import PerAvg
from flcore.trainmodel.models import *
from flcore.trainmodel.transformer import *
from utils.mem_utils import MemReporter
from sklearn import metrics
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635   #98635 for AG_News and 399198 for Sogou_News
max_len=200
emb_dim=32
# 加载模型
support_listX=[]
support_listY=[]
predict=[]
true=[]
# filename="CICIDS2017QuantityFeature3"
filename='TON-IoT'

def matrix_test(y_really,y_predict):
    reporter = MemReporter()
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
    # 计算Precision、Recall和F1 Score,输出array
    matrix = metrics.confusion_matrix(y_really, y_predict, labels=labels)
    # 计算精确度（Precision）
    precision = precision_score(y_really, y_predict, average='macro')
    # 计算召回率（Recall）
    recall = recall_score(y_really, y_predict, average='macro')
    # 计算 F1 值（F1 score）
    f1 = f1_score(y_really, y_predict, average='macro')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("confusion matrix: \n", matrix)

    return precision, recall, f1, matrix

    print("All done!")

    reporter.report()

def run(args,x,y,filename):
    # Conv1d的通道数是指词向量的维度，Conv2d的通道数是指颜色通道
    # 比如：黑白图的通道数是1和RGB彩色图的通道数为3或者设置更多的颜色通道数
    # 几个类别几个dim
    test_acc = 0
    test_num = 0
    y_really=[]
    y_predict=[]
    # 聚合相似度最高的三个模型
    # model0 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 0.pt')
    # model1 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 1.pt')
    # model2 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 2.pt')
    # model3 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 3.pt')
    # model4 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 4.pt')
    # model5 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 5.pt')
    # model6 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 6.pt')
    # model7 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 7.pt')
    # model8 = torch.load(f'../dataforsimi/{filename}/client models/CICIDS2017/PerAvg_client 8.pt')


    model0 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 0.pt')
    model1 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 1.pt')
    model2 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 2.pt')
    model3 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 3.pt')
    model4 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 4.pt')
    model5 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 5.pt')
    model6 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 6.pt')
    model7 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 7.pt')
    model8 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 8.pt')
    model9 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 9.pt')
    model10 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 10.pt')
    model11 = torch.load(f'../dataforsimi/{filename}/client models/PerAvg_client 11.pt')
    new_state_dict = {}
    for key in model1.keys():
        new_state_dict[key] = (model0[key]+model1[key] + model2[key] + model3[key]+model4[key]+model5[key]+model6[key]+model7[key]+model8[key]+model9[key]+model10[key])+model11[key]/11
    with torch.no_grad():
        args.model = FedAvgCNN_conv1(in_channels=25, num_classes=args.num_classes, dim=args.num_classes)
        print(args.model)
        args.model.load_state_dict(new_state_dict)
        args.model.eval()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        output = args.model(x)
        max_indices = (torch.argmax(output, dim=1)).numpy()
        test_acc +=np.sum(max_indices==y)
        test_num += y.shape[0]
        y_predict.append(max_indices)
        y_really.append(y)
    predict.extend(y_predict[0])
    true.extend(y_really[0])
    print("Accurancy:", test_acc / test_num)
    precision, recall, f1, matrix = matrix_test(y_really[0],y_predict[0])
    return y_predict,test_acc,test_num,precision,recall,f1,matrix


# 读取0.npz到8.npz的support文件

parser = argparse.ArgumentParser()
# general
parser.add_argument('-go', "--goal", type=str, default="test",
                    help="The goal for this experiment")
parser.add_argument('-dev', "--device", type=str, default="cuda",
                    choices=["cpu", "cuda"])
parser.add_argument('-did', "--device_id", type=str, default="0")
parser.add_argument('-data', "--dataset", type=str, default="CICIDS2017")
#parser.add_argument('-data', "--dataset", type=str, default="mnist")
parser.add_argument('-nb', "--num_classes", type=int, default=9)
parser.add_argument('-m', "--model", type=str, default="cnn")
parser.add_argument('-lbs', "--batch_size", type=int, default=40)
parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                    help="Local learning rate")
parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
parser.add_argument('-gr', "--global_rounds", type=int, default=100)
parser.add_argument('-ls', "--local_epochs", type=int, default=3,
                    help="Multiple update steps in one local epoch.")
parser.add_argument('-algo', "--algorithm", type=str, default="PerAvg")
parser.add_argument('-jr', "--join_ratio", type=float, default=0.2,
                    help="Ratio of clients per round")
parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                    help="Random ratio of clients per round")
parser.add_argument('-nc', "--num_clients", type=int, default=12,
                    help="Total number of clients")
parser.add_argument('-pv', "--prev", type=int, default=0,
                    help="Previous Running times")
parser.add_argument('-t', "--times", type=int, default=1,
                    help="Running times")
parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                    help="Rounds gap for evaluation")
parser.add_argument('-dp', "--privacy", type=bool, default=False,
                    help="differential privacy")
parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
parser.add_argument('-ab', "--auto_break", type=bool, default=False)
parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
# practical
parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                    help="Rate for clients that train but drop out")
parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                    help="The rate for slow clients when training locally")
parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                    help="The rate for slow clients when sending global model")
parser.add_argument('-ts', "--time_select", type=bool, default=False,
                    help="Whether to group and select clients at each round according to time cost")
parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                    help="The threthold for droping slow clients")
# pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
parser.add_argument('-bt', "--beta", type=float, default=0.001)
parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                    help="Regularization weight")
parser.add_argument('-mu', "--mu", type=float, default=0.0)
parser.add_argument('-K', "--K", type=int, default=5,
                    help="Number of personalized training steps for pFedMe")
parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                    help="personalized learning rate to caculate theta aproximately using K steps")
# FedFomo
parser.add_argument('-M', "--M", type=int, default=5,
                    help="Server only sends M client models to one client at each round")
# FedMTL
parser.add_argument('-itk', "--itk", type=int, default=4000,
                    help="The iterations for solving quadratic subproblems")
# FedAMP
parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                    help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
parser.add_argument('-sg', "--sigma", type=float, default=1.0)
# APFL
parser.add_argument('-al', "--alpha", type=float, default=1.0)
# Ditto / FedRep
parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
# MOON
parser.add_argument('-tau', "--tau", type=float, default=1.0)
# FedBABU
parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
# APPLE
parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
parser.add_argument('-L', "--L", type=float, default=1.0)
# FedGen
parser.add_argument('-nd', "--noise_dim", type=int, default=512)
parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
parser.add_argument('-se', "--server_epochs", type=int, default=1000)
parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
# SCAFFOLD / FedGH
parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
# FedALA
parser.add_argument('-et', "--eta", type=float, default=1.0)
parser.add_argument('-s', "--rand_percent", type=int, default=80)
parser.add_argument('-p', "--layer_idx", type=int, default=2,
                    help="More fine-graind than its original paper.")
# FedKD
parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
parser.add_argument('-Te', "--T_end", type=float, default=0.98)
# FedAvgDBE
parser.add_argument('-mo', "--momentum", type=float, default=0.1)
parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)
args = parser.parse_args()

test_Sdata = []
for i in range(12):
    print(f"===================client{i}==================")
    xfile_name = f"../dataforsimi/{filename}/Stest/{i}x.csv"
    x_np = np.loadtxt(xfile_name,delimiter=',')
    x = torch.Tensor(x_np).type(torch.float32)  # 以示例为准，根据你的模型输入进行设置
    yfile_name = f"../dataforsimi/{filename}/Stest/{i}y.csv"
    y = np.loadtxt(yfile_name, delimiter=',')
    support_listX.append(x)
    support_listY.append(y)
    # 进行预测

    y_predict,test_acc,test_num,precision,recall,f1,matrix=run(args,x,y,filename)
    y_np=y_predict[0]
    # 将 x 和 y_np 放入一个字典
    data_dict = {'x': x_np, 'y': y_np}
    test_Sdata.append(data_dict)

test_Sdata=tuple(test_Sdata)
#将预测的support与原本的support x拼接并存储
for idx, test_Sdict in enumerate(test_Sdata):
    with open(f'../dataforsimi/{filename}/Stest/centerPrelabel/'+ str(idx) + '.npz', 'wb') as f:
        np.savez_compressed(f, data=test_Sdict)


print("all matrix")
matrix_test(true,predict)
cm = confusion_matrix(true, predict)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g',
            xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7',
                         'Class 8'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7',
                         'Class 8'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Use center model for pre-label')
plt.show()