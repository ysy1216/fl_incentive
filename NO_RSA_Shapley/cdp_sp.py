import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
from fl_tools import *
import pickle
import os


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

    train_loader = DataLoader(train_data, batch_size=2000, shuffle=True) #6w/b
    server_train_loader = DataLoader(train_data, batch_size=64, shuffle=True) #6w/b
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False) #1w/b

    return server_train_loader,train_loader, test_loader

# 训练客户端
def client_train(local_model,loss_func,device,optimizer, train_loader):
    local_model.train()
    for i,(x,y) in enumerate(train_loader):
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        loss = loss_func(local_model(x), y)
        loss.backward()
        optimizer.step()
    return local_model,loss


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





def clip_and_noise_gradients(grads, x, clip_ratio, noise_scale, global_model, optimizer):

    # 随机选择裁剪的梯度
    indices = np.random.choice(len(grads), size=x, replace=False)

    # 备份裁剪后的梯度
    clipped_grads = []
    for index in indices:
        grad = grads[index]
        torch.nn.utils.clip_grad_norm_(grad, clip_ratio)
        clipped_grads.append(grad)

    # 将裁剪后的梯度替换回原有位置
    for i, index in enumerate(indices):
        grads[index] = clipped_grads[i]
    
    #得到一个梯度列表 进行平均聚合 需要考虑每个梯度的维度不一致，也就是每一层网络的维度不一致

    layer_grads = []        
    for g in grad:   
        layer_grads.append(g.mean())
        
    average_grad = sum(layer_grads) / len(layer_grads)
    print(average_grad)
    # 对于平局梯度添加差分隐私噪声 - 拉普拉斯机制
    laplace_noise = np.random.laplace(0, clip_ratio / noise_scale, size=average_grad.shape)
    # 添加噪声到梯度中
    private_grad = average_grad + torch.from_numpy(laplace_noise)

 

 
    # 更新全局模型的参数
    for i, params in enumerate(global_model.parameters()):  
        if params.grad is None:
            params.grad = torch.zeros_like(params.data)        
        params.grad.data.copy_(private_grad.data)   
    optimizer.step()    
    optimizer.zero_grad()

    return private_grad, global_model

 


def main():
    for epsilon in range(5, 21, 5):
        epsilon = round(epsilon / 10, 1)
        for ci  in range(1,10):
            epsilon = 0.5        # 隐私预算
            lr=0.1
            epoches=1
            num_clients=60000
            loss_func=nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            global_model = SampleConvNet().to(device)
            server_optimizer = optim.SGD(global_model.parameters(), lr=lr)
            client_models = [SampleConvNet().to(device) for _ in range(num_clients)]
            client_optimizers = [optim.SGD(model.parameters(), lr=lr) for model in client_models]
            server_train_loader,train_loader, test_loader= get_data_loaders()

            for epoch in range(epoches):
                print(f"Round {epoch}/{epoches}")
                grads = []  #记录所有客户端的总梯度 
                for id in range(num_clients):
                    model,loss=client_train(client_models[id],loss_func,device, client_optimizers[id], train_loader) #第id个本地模型训练后的模型和损失
                    print(f'第{epoch}轮次中第{id}个客户端在该轮次的train_loss为{loss}')
                    #收集本轮次中该客户端的本地梯度 
                    grads.append([param.grad.clone() for param in model.parameters()])

                #测试客户端
                acces=[]
                for id in range(num_clients):
                    acc=test_model(client_models[id],device,test_loader)
                    acces.append(acc)
                    print(f'第{epoch}轮次中第{id}的客户端在测试集的精度为{acc}')
                #求shapley值

                acces=np.array(acces)
                shapley_values=parallel_monte_carlo_shapley(num_samples, acces, evaluate)
                #当出现负值的shapley值，我们认定该值是无效客户端或者恶意客户端表现出的来性能，我们直接设为0
                shapley_values=np.array(shapley_values)
                shapley_values[shapley_values<=0]=0
                shapley_values/=shapley_values.sum()
                shapley_values=list(shapley_values)

                file_path = os.path.join('./result/CDP/', f'shapley_CDP_{epsilon}_{ci}.pkl')  # 文件路径为 result/sp_ldp1_{epsilon}_1.pkl

                with open(file_path, 'wb') as f:
                    pickle.dump(shapley_values, f)
                
                #梯度处理，加入拉普拉斯噪声并随机梯度裁剪 裁剪比例为1/100，挑选数量比例为1/6
                #1,收集所有梯度再处理（进行中心化差分隐私机制）
                print('开始梯度加噪并裁剪处理') 
                grads_cdp,global_model=clip_and_noise_gradients(grads, x=10000, clip_ratio=1/100, noise_scale=epsilon,global_model=global_model,optimizer=server_optimizer)

                os.makedirs('./result/', exist_ok=True)
                file_path = os.path.join('./result/CDP/', f'grads_CDP_{epsilon}_{ci}.pkl')  # 文件路径为 result/gards_cdp1_{epsilon}_1.pkl
                with open(file_path, 'wb') as f:
                    pickle.dump(grads_cdp, f)
            
                print('聚合完成,开始全局模型更新')
                # 训练全局模型
                global_model,loss=server_train(global_model,loss_func,device, server_optimizer,server_train_loader)
                print(f'服务器在第{epoch}轮次的loss为{loss}')

                torch.save(global_model.state_dict(), 'global_model_params_cdp.pth')
                l_model = SampleConvNet().to(device)  
                # 加载全局模型参数
                global_model_params = torch.load('global_model_params_cdp.pth')
                l_model.load_state_dict(global_model_params)

                client_models = [l_model for _ in range(num_clients)]
                #测试全局模型
                s_test_acces=test_model(global_model,device,test_loader)
                print(f'在第{epoch}轮次中server测试精度为{s_test_acces}%')

if __name__ == '__main__':
    main()

