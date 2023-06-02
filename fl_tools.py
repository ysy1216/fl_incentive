import random
from itertools import combinations, permutations
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time

def partial_accuracy(acc_list, S):
    """
    计算指定组合情况下的集成模型预测精度
    
    Args:
        acc_list(list): 预测结果精度列表
        S(set): 指定组合，其中包含参与者在内的一部分元素
        
    Returns:
        float: 集成模型的预测精度
    """
    precision = 0
    for i in S:
        precision += acc_list[i]
    
    return precision / len(S)

def compute_shapley_value(acc_list, default_weights=None):
    """
    计算指定精度列表的Shapley Value
    
    Args:
        acc_list(list): 预测结果精度列表
        default_weights(list): 参与者默认权重值
    
    Returns:
        list: 各参与者的Shapley value
    """
    n = len(acc_list)
    
    # 如果未指定权重值，则使用默认值1/n
    if default_weights is None:
        default_weights = [1/n] * n
        
    shapley_values = [0]*n
    
    for i in range(n):
        p = 0  # 记录每个参与者的预计总价值
        
        # 枚举包含当前参与者i的子集，并计算对应的贡献值
        for k in range(n):
            for random_subset in combinations(set(range(n)) - {k}, k):
                if i in random_subset:
                    precision_with_i = partial_accuracy(acc_list, random_subset + (i,))
                    precision_without_i = partial_accuracy(acc_list, random_subset)
                    delta = precision_with_i - precision_without_i
                    p += delta
                    
        phi = p / ((n-1)*2**(n-2))
        
        # 根据Shapley value定义计算当前参与者的Shapley value
        shapley_values[i] = sum([choose(len(random_subset), k) *
                                 (p - phi) for k in range(len(random_subset)+1)])
        
    return shapley_values

def choose(n, k):
    """
    计算组合数C(n,k)
    
    Args:
        n(int): 组合总数
        k(int): 每组数字的个数
    
    Returns:
        int: 组合数结果
    """
    if k > n or n < 0 or k < 0:
        return 0
    elif k == 0 or k == n:
        return 1

    numerator = 1
    denominator = 1
    for i in range(1, min(k, n-k)+1):
        numerator *= n+1-i
        denominator *= i

    return numerator // denominator


# 测试用例


# 当出现负值的shapley值，我们认定该值是无效客户端或者恶意客户端表现出的来性能，我们直接设为0
# shapley_values=np.array(shapley_values)
# shapley_values[shapley_values<=0]=0
# shapley_values/=shapley_values.sum()
# shapley_values=list(shapley_values)
# print(shapley_values)
a=time.time()
# 测试用例
lst = [0.9, 0.85, 0.75, 0.7, 0.6,9]
shapley_values = compute_shapley_value(lst)
print(shapley_values)
print('time6',time.time()-a)

a=time.time()
# 测试用例
lst = [0.9, 0.85, 0.75, 0.7, 0.6,1,2,3,6]
shapley_values = compute_shapley_value(lst)
print(shapley_values)
print('time9',time.time()-a)

a=time.time()
# 测试用例
lst = [0.9, 0.85, 0.75, 0.7, 0.6,1,2,3,6,0.85, 0.75, 0.7,0.9, 0.85, 0.75, 0.7, 0.6,1,2,3,6,0.85, 0.75, 0.7]
shapley_values = compute_shapley_value(lst)
print(shapley_values)
print('time12',time.time()-a)

