import torch
import numpy as np

# 模拟4个客户端并存储每个客户端的梯度向量
grad1 = torch.tensor([0.5, 0.3, -0.2, 0.1])
grad2 = torch.tensor([-0.2, 0.4, 0.1, 0.3])
grad3 = torch.tensor([0.1, 0.2, 0.3, 0.4])
grad4 = torch.tensor([0.3, -0.1, 0.4, -0.2])

num_clients = 4
client_gradients = [grad1, grad2, grad3, grad4]

# 计算平均梯度向量
average_gradient = sum(client_gradients) / num_clients

# 计算所有可能的Coalition (2**num_clients - 1)
coalitions = []
for i in range(1, 2 ** num_clients):
    bin_str = bin(i)[2:].zfill(num_clients)
    coalitions.append(np.array([int(c) for c in bin_str]))

# 计算每个Coalition下的贡献
shapley_values = [0] * num_clients
for c in coalitions:
    total = 0
    for i in range(num_clients):
        if c[i] == 1:
            # 计算加入当前Coalition对应的子集后的所有梯度的平均值
            coalition_grads = [client_gradients[j] for j in range(num_clients) if c[j] == 1]
            coalition_average_gradient = sum(coalition_grads) / sum(c)
            
            # 计算加入当前客户端对Coalition的贡献(runtime difference)并累加到当前Coalition的总共计数( 对每个客户都是不同的)
            marginal_contribution = (coalition_average_gradient - total) / sum(c)
            shapley_values[i] += marginal_contribution
            
            # 更新总计数（runs to update）
            total += marginal_contribution * sum(c)

# 对客户端梯度进行重新加权，并更新全局模型参数
weights = torch.tensor([0.2, 0.3, 0.4, 0.1])
weighted_grads = torch.stack(client_gradients) * weights[:, None]

# 特别注意，我们这里的平均梯度向量是使用所以客户端进行求解得出
reweighted_average_grad = sum(weighted_grads) / num_clients

# 在例子中直接输出结果，将其替换为实际需要手动处理的部分
print("Average gradient vector:", average_gradient)
print("Shapley values:", shapley_values)
print("Reweighted average gradient vector:", reweighted_average_grad)



