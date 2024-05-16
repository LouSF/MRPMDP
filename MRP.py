
"""
1. 马尔可夫奖励过程建立
2. 回报值计算
3. 状态价值计算
"""


import numpy as np


"""
马尔可夫奖励过程建立：
    - 初始化状态字典
    - 初始化状态转移概率矩阵
    - 初始化奖励函数
"""
num_states = 7
# 索引到状态名的字典
i_to_n={}
i_to_n["0"] = "C1"
i_to_n["1"] = "C2"
i_to_n["2"] = "C3"
i_to_n["3"] = "Pass"
i_to_n["4"] = "Pub"
i_to_n["5"] = "FB"
i_to_n["6"] = "Sleep"


# 状态名到索引的字典
n_to_i={}
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

# 状态转移概率矩阵
Pss=[
    [ 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0 ],
    [ 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2 ],
    [ 0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0 ],
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ],
    [ 0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0 ],
    [ 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0 ],
    [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 ]
]
Pss = np.array(Pss)

#奖励函数，奖励与状态相对应
rewards = [-2, -2, -2, 10, 1, -1, 0]


def compute_return(start_index,chain,gamma):
    """
    计算一个马尔科夫奖励过程中某状态的收获值
    :param start_index: 要计算的状态在链中的位置
    :param chain: 要计算的马尔可夫过程(马尔可夫链)
    :param gamma: 衰减因子
    :return: 收获值
    """
    retrn, power, gamma = 0.0, 0, gamma
    for i in range(start_index, len(chain)):
        retrn += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
        power += 1
    return retrn


chains = [
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB","FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

retrn = compute_return(start_index=0,chain= chains[3],gamma = 0.5)
print("对于马尔可夫过程:",chains[3],"\n获得的回报为:",retrn)


def compute_value(Pss,rewards,gamma):
    """
    通过求解矩阵方程的形式直接计算状态的价值
    :param Pss: 状态转移概率矩阵；shape(7,7)
    :param rewards: 即时奖励 list
    :param gamma: 衰减因子
    :return: 各状态的价值
    """

    # 将rewards转为numpy数组并修改为列向量的形式
    rewards = np.array(rewards).reshape((-1, 1))
    # np.eye(7,7)为单位矩阵，inv方法为求矩阵的逆
    values = np.dot(np.linalg.inv(np.eye(7, 7) - gamma * Pss), rewards)
    return values

values = compute_value(Pss, rewards, gamma = 0.99999)
print("\n各状态下的价值函数为:\n",values)
