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

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True) #6w/b
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




def generate_grads_with_privacy_ldp(grads, num_selected, clip_norm, epsilon):
    """
    将梯度列表添加本地差分隐私，并对部分梯度进行随机裁剪
    参数:
    - grads: 梯度列表，包含多个梯度张量
    - num_selected: 随机裁剪的梯度数量
    - clip_norm: 裁剪比例，用于控制梯度裁剪的范围
    - epsilon: 隐私预算，控制添加的差分隐私噪声大小
    返回:
    - private_grads: 添加了本地差分隐私，并进行随机裁剪的梯度列表，包含原始梯度和裁剪梯度
    """
    # 将梯度列表转换为张量形式  
    grads = [torch.stack(g) for g in zip(*grads)]
    # 添加差分隐私噪声
    private_grads = []
    for i in range(len(grads)):
        # 生成服从拉普拉斯分布的随机噪声
        laplace_noise = np.random.laplace(0, clip_norm / epsilon, size=grads[i].shape)
        # 添加噪声到梯度中
        private_grad = grads[i] + torch.from_numpy(laplace_noise).to(grads[i].device)
        private_grads.append(private_grad)

    # 随机选择裁剪的梯度
    selected_indices = np.random.choice(len(private_grads), size=num_selected, replace=False)

    # 对部分梯度进行裁剪
    for i in selected_indices:
        total_norm = torch.norm(private_grads[i].detach())  # 计算梯度张量的范数
        clip_coef = clip_norm / (total_norm + 1e-6)  # 计算裁剪比例
        private_grads[i] = private_grads[i] * clip_coef  # 对梯度进行裁剪
    # 将张量形式的梯度列表转换为普通梯度列表
    private_grads= [[param.clone().detach() for param in model] for model in zip(*private_grads)]
    return private_grads




def shapley_juhe(global_model, optimizer, shuffle_list):
    # 遍历全局模型的参数
    for i, params in enumerate(global_model.parameters()):
        if params.grad is None:
            continue
        # 初始化全局梯度为零
        global_grad = torch.zeros_like(params.grad)
        # 根据 Shapley 权重计算加权平均梯度
        for j, pair in enumerate(shuffle_list):
            local_grad, shapley_weight = pair
            global_grad += local_grad[i] * shapley_weight
        # 更新全局模型参数的梯度
        params.grad = global_grad
    # 使用优化器执行梯度下降更新
    optimizer.step()
    global_model.zero_grad()
    return global_model


def main():
    for epsilon in range(5, 21, 5):
        epsilon = round(epsilon / 10, 1)
        for ci in range(1,10):
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

            #开始训练
            '''
            每个轮次，60000个客户端训练,针对60000个模型的梯度加入拉普拉斯噪声，随机挑选10000梯度裁剪，裁剪比例是1/100，测试各个6万个模型的acc，针对acc列表进行shapley值，对应聚合全局模型，训练全局模型并测试。
            '''
            for epoch in range(epoches):
                print(f"Round {epoch}/{epoches}")
                #训练客户端
                grads = []  #记录所有客户端的总梯度 
                for id in range(num_clients):
                    model,loss=client_train(client_models[id],loss_func,device, client_optimizers[id], train_loader) #第id个本地模型训练后的模型和损失
                    print(f'第{epoch}轮次中第{id}个客户端在该轮次的train_loss为{loss}')
                    #收集本轮次中该客户端的本地梯度 
                    grads.append([param.grad.clone() for param in model.parameters()])
                #梯度处理，加入拉普拉斯噪声并随机梯度裁剪 裁剪比例为1/100，挑选数量比例为1/6
                #1,收集所有梯度再处理（进行中心化差分隐私机制）
                print('开始梯度加噪并裁剪处理') 
                # grads_cdp=generate_grads_with_privacy_cdp(grads, num_selected=166,clip_norm=1/100, epsilon=1.5,device=device)
                # grads=grads_cdp
                #2,针对每一个梯度加入拉普拉斯噪声，（进行本地化差分隐私机制）
                grads_ldp=generate_grads_with_privacy_ldp(grads, num_selected=10000, clip_norm=1/100, epsilon=epsilon)
                grads=grads_ldp
                #3，真实梯度不处理
                # gards=gards
            

                os.makedirs('./result/', exist_ok=True)
                file_path = os.path.join('./result/LDP/', f'grads_LDP_{epsilon}_{ci}.pkl')  # 文件路径为 result/gards_ldp1_{epsilon}_1.pkl

                with open(file_path, 'wb') as f:
                    pickle.dump(grads, f)


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
                
                file_path = os.path.join('./result/LDP/', f'shapley_LDP_{epsilon}_{ci}.pkl')  # 文件路径为 result/sp_ldp1_{epsilon}_1.pkl

                with open(file_path, 'wb') as f:
                    pickle.dump(shapley_values, f)


                #随机打乱
                shuffle__list = [[x, y] for x, y in zip(grads, shapley_values)]
                random.shuffle(shuffle__list)
                print('开始聚合')
                #利用shapley值作为各个梯度之间的权重关系更新全局模型
                global_model=shapley_juhe(global_model,server_optimizer,shuffle__list)
                print('聚合完成,开始全局模型更新')
                # 训练全局模型
                global_model,loss=server_train(global_model,loss_func,device, server_optimizer,server_train_loader)
                print(f'服务器在第{epoch}轮次的loss为{loss}')

                torch.save(global_model.state_dict(), 'global_model_params_ldp.pth')
                l_model = SampleConvNet().to(device)  
                # 加载全局模型参数
                global_model_params = torch.load('global_model_params_ldp.pth')
                l_model.load_state_dict(global_model_params)

                client_models = [l_model for _ in range(num_clients)]
                #测试全局模型
                s_test_acces=test_model(global_model,device,test_loader)
                print(f'在第{epoch}轮次中server测试精度为{s_test_acces}%')

if __name__ == '__main__':
    main()

