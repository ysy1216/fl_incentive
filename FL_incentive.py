import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import torch.nn.functional as F
import torch.optim as optim
import opacus 
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torchvision import datasets, transforms
from tqdm import tqdm
import math
import itertools
import random
import copy
from fl_tools import *
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

    train_loader = DataLoader(train_data, batch_size=3000, shuffle=True) #6w/b
    test_loader = DataLoader(test_data, batch_size=200, shuffle=False) #1w/b

    return train_loader, test_loader

# 训练客户端
def client_train(local_model,loss_func,device,optimizer, train_loader):
    start=time.time()
    local_model=nn.DataParallel(local_model)
    local_model.train()
    for i,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        loss = loss_func(local_model(x), y)
        loss.backward()
        optimizer.step()
    time1=time.time()-start
    return time1,local_model,loss


# 训练服务端  
def server_train(model,loss_func,device,optimizer, data_loader):
    # model=nn.DataParallel(model)
    model.train()
    for i,(x,y) in enumerate(data_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        loss = loss_func(model(x), y)
        loss.backward()
        optimizer.step()
    return model,loss
      



# 测试模型
def test_model(model,device,test_loader):
    #返回所有客户端测试的精度统计列表
    model.eval()
    total,correct=0,0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            x,y=x.to(device),y.to(device)
            pred=model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy




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


def generate_grads_with_privacy(grads, num_selected, clip_norm, epsilon, device):
    # 将梯度列表转换为张量形式
    grads_tensor = [torch.stack(g) for g in zip(*grads)]

    # 加入拉普拉斯噪声
    noise_scale = clip_norm / epsilon  # 噪声缩放因子
    noisy_grads_tensor = []
    for g in grads_tensor:
        laplace_noise = torch.tensor(np.random.laplace(0, noise_scale, g.shape)).to(device=device)
        noisy_grads_tensor.append(g + laplace_noise)
    
    # 随机挑选客户端
    
    selected_indices = np.random.choice(len(grads), num_selected, replace=False)
    
    # 裁剪所选客户端的梯度
    clipped_grads_tensor = []
    for i in selected_indices:
        client_grads_tensor = [noisy_grads_tensor[j][i] for j in range(len(grads_tensor))]
        clipped_client_grads_tensor = [torch.clamp(g, -clip_norm, clip_norm) for g in client_grads_tensor]
        clipped_grads_tensor.append(clipped_client_grads_tensor)
    
    # 将裁剪后的梯度替换回原梯度
    new_grads_tensor = [g.clone() for g in grads_tensor]
    for i, idx in enumerate(selected_indices):
        for j, g in enumerate(clipped_grads_tensor[i]):
            new_grads_tensor[j][idx] = g
    
    # 将张量形式的梯度列表转换为普通梯度列表
    new_grads = [[param.clone().detach() for param in model] for model in zip(*new_grads_tensor)]
    return new_grads



def shapley_juhe(global_model, optimizer, local_grads, shapley_weights):
    print(f'聚合比列表sapley_weights:{shapley_weights}')
    # 计算加权平均梯度
    local_grads=local_grads.to(device='cpu')
    mean_grads = np.mean([np.array(grad) * weight for grad, weight in zip(local_grads, shapley_weights)], axis=0)

    # 更新全局模型
    for param, grad in zip(global_model.parameters(), mean_grads):
        if grad is not None:
            param.grad = torch.from_numpy(grad).to(device=param.device)
    optimizer.step()
    global_model.zero_grad()
    return global_model


def main():
    # alpha = 1/100        # 梯度裁剪比例
    # epsilon = 1.5        # 隐私预算
    lr=0.3121
    epoches=10
    num_clients=2
    # cur_c_num=10000
    # privacy_engine = opacus.PrivacyEngine()
    loss_func=nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = SampleConvNet().to(device)
    server_optimizer = optim.SGD(global_model.parameters(), lr=lr)
    client_models = [SampleConvNet().to(device) for _ in range(num_clients)]
    client_optimizers = [optim.SGD(model.parameters(), lr=lr) for model in client_models]
    train_loader, test_loader= get_data_loaders()

    #开始训练
    '''
    每个轮次，60000个客户端训练,针对60000个模型的梯度加入拉普拉斯噪声，随机挑选10000梯度裁剪，裁剪比例是1/100，测试各个6万个模型的acc，针对acc列表进行shapley值，对应聚合全局模型，训练全局模型并测试。
    '''
    for epoch in range(epoches):
        print(f"Round {epoch}/{epoches}")
        #训练客户端
        grads = []  #记录所有客户端的总梯度用于加噪裁剪和计算对应shapley值
        for id in range(num_clients):
            time1,model,loss=client_train(client_models[id],loss_func,device, client_optimizers[id], train_loader) #第id个本地模型训练后的模型和损失
            print(f'第{epoch}轮次中第{id}个客户端在该轮次的train_loss为{loss},花费{time1}秒')
            #收集本轮次中该客户端的本地梯度
            grads.append([param.grad.clone() for param in model.parameters()])
        #梯度处理，加入拉普拉斯噪声并随机梯度裁剪
        print('开始梯度加噪并裁剪处理')
        grads=generate_grads_with_privacy(grads, num_selected=1, clip_norm=1.0/100.0, epsilon=1.5,device=device)
        print('开始测试客户端')
        #测试客户端
        acces=[]
        for id in range(num_clients):
            acc=test_model(client_models[id],device,test_loader)
            acces.append(acc)
            print(f'第{epoch}轮次中第{id}的客户端在测试集的精度为{acc}')
        #求shapley值
        #保留测试的mse值 进行shapley计算 ，但越高的mse贡献越大，所以采用acc
        #基于mse的shapley值计算//基于acc的shpaley值计算
        acces=np.array(acces)
        shapley_values=parallel_monte_carlo_shapley(num_samples, acces, evaluate)
        #当出现负值的shapley值，我们认定该值是无效客户端或者恶意客户端表现出的来性能，我们直接设为0
        shapley_values=np.array(shapley_values)
        shapley_values[shapley_values<=0]=0
        shapley_values/=shapley_values.sum()
        shapley_values=list(shapley_values)
        # print(shapley_values)


        #？？  i,根据训练后全局模型计算出各个梯度的贡献值，下一轮的时候着重考虑贡献值高的梯度
        #？？  ii，根据测试loss进行各个梯度的贡献值，立刻调整全局模型的梯度后进行训练及测试
        # ii
        print('开始聚合')
        #利用shapley值作为各个梯度之间的权重关系更新全局模型
        global_model=shapley_juhe(global_model,server_optimizer,grads,shapley_values)
        print('聚合完成,开始全局模型更新')
        # 训练全局模型
        global_model,loss=server_train(global_model,loss_func,device, server_optimizer,train_loader)
        print(f'服务器在第{epoch}轮次的loss为{loss}')
 
        # #服务器发送训练后的全局模型参数
        # l_model=SampleConvNet().to(device)
        # l_model.load_state_dict(global_model.state_dict())
        # client_models = [l_model for _ in range(num_clients)]
        #若采用分布式训练，修改模型
        l_model = SampleConvNet().to(device)
        global_model_state_dict = global_model.state_dict()
        if 'module.' in list(global_model_state_dict.keys())[0]: # 如果模型参数中包含 "module." 前缀的名称空间
            global_model_state_dict = {k.replace('module.', ''): v for k, v in global_model_state_dict.items()}
        l_model.load_state_dict(global_model_state_dict)
        client_models = [l_model for _ in range(num_clients)]

        #测试全局模型
        s_test_acces=test_model(global_model,device,test_loader)
        print(f'在第{epoch}轮次中server测试精度为{s_test_acces}%')

if __name__ == '__main__':
    main()

