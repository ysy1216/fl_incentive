import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision import datasets, transforms
from tqdm import tqdm
import math
import itertools
import random
import copy
'''
CLDP-SGD Model Architecture for MNIST
Layer               Parameters
Convolution         16 filters of 8 × 8, Stride 2
Max-Pooling         2 × 2
Convolution         32 filters of 4 × 4, Stride 2
Max-Pooling         2 × 2
Fully connected     32 unites
Softmax             10 unites
Runs MNIST training with differential privacy.
'''
#模型
class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]

        return x


# 载入数据
def get_data_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    return train_loader, test_loader,train_data,test_data

# 训练客户端
def client_train(local_model,loss_func,device,optimizer, data,target):
    local_model.train()
    local_model.to(device)
    data = torch.tensor(data, dtype=torch.float32,device=device)
    target = torch.tensor(target, dtype=torch.float32,device=device)
    optimizer.zero_grad()
    output = local_model(data)
    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
  
    return local_model,loss


# 训练服务端  
def server_train(model,loss_func,device, optimizer, data,target):
    model.train()
    model.to(device)
    data= torch.tensor(data, dtype=torch.float32,device=device)
    target=torch.tensor(target, dtype=torch.float32,device=device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_func(output, target)
    loss.backward()
    optimizer.step()
      
    return model,loss



# 测试模型
def test_model(model, loss_func, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output,target.unsqueeze(1)).item() # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    return test_loss,correct / len(test_loader.dataset)


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


#clients_gradients是一个元素大小为10的列表，每一个元素包含一个OrderedDict。
#举例：        print(local_gradients[0])
    # OrderedDict([('conv1.weight',tensor([[[[ 6.4584e-02,...,]]]], device='cuda:3')),
    #             ('conv1.bias', tensor([-0.0973,...],device='cuda:3')), 
    #             ('conv2.weight', tensor([[[[ 0.0027,...]]]], device='cuda:3')), 
    #             ('conv2.bias', tensor([ 0.0080,...],device='cuda:3')), 
    #             ('fc1.weight', tensor([[ 0.0190,...]],device='cuda:3')), 
    #             ('fc1.bias', tensor([-1.1140e-02,...], device='cuda:3')), 
    #             ('fc2.weight', tensor([[ 0.0041,...]],device='cuda:3')), 
    #             ('fc2.bias', tensor([-0.0555,...], device='cuda:3'))])
# 每一个OrderedDict中包含8个('layer_name',tensor(梯度)):包含conv1.weight层，conv1.bias层，conv2.weight层，conv2.bias层，fc1.weight层 fc.bias层，fc2.weight层 fc2.bias层 共8个
#从client_gradients中剥离出相同层的梯度,然后输出格式为[[tensor1,tensor2...tensor10],[tensor1,tensor2..tensor10]...[]]
#输出的layers_list的格式为8*10 表示8层中10个客户端的梯度 剥离出8*len(client_gradindts)的大小列表

# def boli(client_gradients):
#     """
#     从 client_gradients 中剥离出相同层的梯度

#     Args:
#         client_gradients: 大小为 10 的列表，每一个元素包含一个 OrderedDict。
#                           每个 OrderedDict 中包含 8 个 ('layer_name', tensor(梯度)): 包含
#                           conv1.weight 层，conv1.bias 层，conv2.weight 层，conv2.bias 层，
#                           fc1.weight 层 fc.bias 层，fc2.weight 层 fc2.bias 层 共 8 个。

#     Returns:
#         layers_list: 列表，包含每个层的所有客户端的梯度向量集合，格式为[[tensor1,tensor2...tensor10],[tensor1,tensor2..tensor10]...[]]
#     """
#     layers_name= ['conv1.weight','conv1.bias','conv2.weight','conv2.bias','fc1.weight','fc1.bias','fc2.weight','fc2.bias']
#     layers_list = [[] for _ in range(8)] # 初始化每个层的梯度张量集合
#     n=len(client_gradients)
#     for i in range(len(layers_name)):
#         for j in range(len(client_gradients)):
#             for k,v in client_gradients[j].items():
#                 if k==layers_name[i]:
#                     layers_list[i].append([v])
#     return layers_list,len(layers_list[7]),len(layers_list[7][0])  #8*10  8层  10个客户端的每一层

# def compute_shapley_value(layer_gard, weights=None):
#     """
#     计算每个客户端的 Shapley 值和重新加权梯度
#     Args:
#         layer_gard: 所有客户端某一层的所有梯度集合
#         weights: tensor, 客户端的权重，用于加权计算平均梯度向量。默认为均匀分布。
        
#     Returns:
#         shapley_values: list of floats, 每个客户端的 Shapley 值
#         reweighted_average_grad: tensor, 重新加权后的该网络层的平均梯度向量
#     """

#     return shapley_values,the_average_grad  #该层所有元素所占的shapley值，以及根据shapley值更新的该梯度均值

# def pinjie(global_model):
#     #将各个层进行shapley值求加权平均后，加载到全局模型参数上
#     return global_model


# def  sample_data(train_data,num_clients,cur_c_num):
#     #随机抽样cur_c_num个客户端的cur份数据
#     data_idx = random.sample(range(len(train_data)), cur_c_num)
#     client_data = [train_data[i] for i in data_idx] # 10个(x,y)数据点给10个客户端

#     client_ids = list(range(num_clients)) # 所有客户端ID
#     selected_ids = random.sample(client_ids, cur_c_num) # 随机选择10个户端
#     print(f'本轮随机抽取的客户端id是{selected_ids}')
#     client_data = [[id,x,y] for id,(x,y) in zip(selected_ids, client_data)]  #组装[id,x,y]
#     return client_data


def compute_shapley_value(lst, weights=None):
    n = len(lst)
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights)

    shapley_values = np.zeros(n)
    for i in range(n):
        for j in range(1, n+1):
            for combination in itertools.combinations(range(n), j):
                if i in combination:
                    weight_sum = np.sum(weights[list(combination)])
                    marginal_contribution = np.sum(np.array([lst[k][1] for k in combination])) / weight_sum
                    shapley_values[i] += marginal_contribution * math.factorial(n - len(combination)) * math.factorial(len(combination) - 1)

    result = [[lst[i][0], shapley_values[i] / math.factorial(n)] for i in range(n)]
    return result



def shapley_juhe(global_model,optimizer,local_grads,shapley_weights):
    #根据shapley值来更新全局模型
    shapley_weights= [w / sum(shapley_weights) for w in shapley_weights]
    print(shapley_weights)
    for i, params in enumerate(global_model.parameters()):
        if params.grad is None:
            continue
        global_grad = torch.zeros_like(params.grad)            
        for j in range(len(shapley_weights)):
            global_grad += local_grads[j][i] * shapley_weights[j]
        params.grad = global_grad
    optimizer.step()   
    return global_model


def main():
    lr=0.01
    epoches=100
    # num_clients=60
    # cur_c_num=10
    num_clients=60000
    cur_c_num=10000
    loss_func=nn.MSELoss(reduction='mean')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    global_model = SampleConvNet().to(device)
    client_models = [SampleConvNet().to(device) for _ in range(num_clients)]

    train_loader, test_loader ,train_data,test_data = get_data_loaders()


    #开始训练
    for epoch in range(epoches):
        print(f"Round {epoch + 1}/{epoches}")
        #随机抽样cur_c_num个客户端的cur份数据
        data_idx = random.sample(range(len(train_data)), cur_c_num)  #随机选取10个数字
        client_data = [train_data[i] for i in data_idx] # 10个(x,y)数据点给10个客户端

        client_ids = list(range(num_clients)) # 所有客户端ID
        selected_ids = random.sample(client_ids, cur_c_num) # 随机选择10个户端
        print(f'本轮随机抽取的客户端id是{selected_ids}')
        client_data = [[id,x,y] for id,(x,y) in zip(selected_ids, client_data)]  #组装[id,x,y]
        # client_data=sample_data(train_data,num_clients,cur_c_num)
        client_optimizers = [optim.SGD(model.parameters(), lr=lr) for model in client_models]
        
        
        #训练客户端
        local_grads = []  #记录所有客户端的总梯度用于计算对应shapley值
        for id,x,y in client_data:
            cur_model,loss=client_train(client_models[id],loss_func,device, client_optimizers[id], x,y) #第id个本地模型训练后的模型和损失
            print(f'第{id}个客户端在该轮次的train_loss为{loss.item()}')
            #收集本轮次中该客户端的本地梯度
            local_grads.append([param.grad.clone() for param in cur_model.parameters()])


        #测试客户端
        t_data_idx = random.sample(range(len(test_data)), cur_c_num)
        test_train_data= [train_data[i] for i in t_data_idx] # 10个(x,y)数据点给10个客户端
        test_data = [[id,x,y] for id,(x,y) in zip(selected_ids, test_train_data)]
        test_losses=[]
        test_acces=[]
        for idx in selected_ids:
            test_loss,test_acc=test_model(client_models[idx],loss_func,device,test_loader)
            test_losses.append([idx,test_loss])
            test_acces.append([idx,test_acc])

        #保留测试的mse值 进行shapley计算 
        #基于mse的shapley值计算//基于acc的shpaley值计算
        shapley_values=compute_shapley_value(test_acces)# [[id,shapley1],[id2,s2]...]

        # print(shapley_values)
        new_list = [item[1] for item in shapley_values]  #列出shapley值列表，与梯度列表对应
        print(f'该{epoch}轮次中各个客户端依次对应的shapley值是{new_list}')

        #？？  i,根据训练后全局模型计算出各个梯度的贡献值，下一轮的时候着重考虑贡献值高的梯度
        #？？  ii，根据测试loss进行各个梯度的贡献值，立刻调整全局模型的梯度后进行训练及测试
        # ii
        #更新全局模型
        server_optimizer = optim.SGD(global_model.parameters(), lr=lr)
        global_model=shapley_juhe(global_model,server_optimizer,local_grads,new_list)
        
       
        # 训练全局模型
        server_idx=random.sample(range(len(train_data)),1)#1个服务器
        server_data=[train_data[i] for i in server_idx] #随机挑选数据
        for x,y in server_data:
            global_model,loss=server_train(global_model,loss_func,device, server_optimizer,x,y)
            print(f'服务器在第{epoch}轮次的loss为{loss.item()}')
 
        #服务器发送训练后的全局模型参数
        l_model=SampleConvNet().to(device)
        l_model.load_state_dict(global_model.state_dict())
        client_models = [l_model for _ in range(num_clients)]
        #测试全局模型
        loss,acc=test_model(global_model, loss_func,device, test_loader)
        acc*=100
        print(f'在第{epoch}轮次中server的测试损失为{loss},测试精度为{acc}%')

if __name__ == '__main__':
    main()

