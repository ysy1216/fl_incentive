import numpy as np
from multiprocessing import Pool, cpu_count
import time
a=time.time()
def evaluate(subset):
    return np.mean(subset)

def monte_carlo_shapley(num_samples, values_list, evaluate_function):
    n = len(values_list)
    shapley_values = np.zeros(n)
    total_permutations = np.math.factorial(n)

    for _ in range(num_samples):
        permuted_indices = np.random.permutation(n)
        subset_value = 0
        for i in range(n):
            new_subset_value = evaluate_function(values_list[permuted_indices[:i + 1]])
            marginal_contribution = new_subset_value - subset_value
            shapley_values[permuted_indices[i]] += marginal_contribution
            subset_value = new_subset_value

    shapley_values /= num_samples
    return shapley_values

def parallel_monte_carlo_shapley(num_samples, values_list, evaluate_function):
    with Pool(cpu_count()) as p:
        samples_per_process = num_samples // cpu_count()
        args = [(samples_per_process, values_list, evaluate_function) for _ in range(cpu_count())]
        results = p.starmap(monte_carlo_shapley, args)

    #  计算每个元素对应的Shapley值列表
    shapley_values = np.mean(results, axis=0)
    shapley_values_list = shapley_values.tolist()
    return shapley_values_list

num_samples = 5000
# values_list = np.random.randint(0, 101, 6) 
# # print(values_list)
# # values_list=np.array(values_list)
# shapley_values = parallel_monte_carlo_shapley(num_samples, values_list, evaluate)
# # 输出每个元素的Shapley值

# print(shapley_values)

 

