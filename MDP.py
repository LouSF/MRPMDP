# 导入工具函数︰根据状态和行为生成操作相关字典的键，显示字典内容
from utils import str_key, display_dict
# 设置转移概率、奖励值以及读取它们方法
from utils import set_prob, set_reward, get_prob, get_reward
# 设置状态价值、策略概率以及读取它们的方法
from utils import set_value, set_pi, get_value, get_pi


# 构建学生马尔科夫决策过程
S = ['浏览手机中','第一节课','第二节课','第三节课','休息中']
A = ['浏览手机','学习','离开浏览','泡吧','退出学习']
R = {} # 奖励Rsa字典
P = {} # 状态转移概率Pss'a字典
gamma = 1.0 # 衰减因子



set_prob(P, S[0], A[0], S[0])  
set_prob(P, S[0], A[2], S[1])  
set_prob(P, S[1], A[0], S[0])   
set_prob(P, S[1], A[1], S[2])   
set_prob(P, S[2], A[1], S[3]) 
set_prob(P, S[2], A[4], S[4])   
set_prob(P, S[3], A[1], S[4])   
set_prob(P, S[3], A[3], S[1], p = 0.2)   
set_prob(P, S[3], A[3], S[2], p = 0.4)  
set_prob(P, S[3], A[3], S[3], p = 0.4)   

set_reward(R, S[0], A[0], -1) 
set_reward(R, S[0], A[2], 0)  
set_reward(R, S[1], A[0], -1)   
set_reward(R, S[1], A[1], -2)   
set_reward(R, S[2], A[1], -2)  
set_reward(R, S[2], A[4], 0)  
set_reward(R, S[3], A[1], 10)  
set_reward(R, S[3], A[3], 1)   

MDP = (S, A, R, P, gamma)

print("----状态转移概率字典（矩阵)信息∶----")
display_dict(P)
print("----奖励字典(函数)信息∶----")
display_dict(R)

Pi = {}
set_pi(Pi, S[0], A[0], 0.5)  
set_pi(Pi, S[0], A[2], 0.5) 
set_pi(Pi, S[1], A[0], 0.5) 
set_pi(Pi, S[1], A[1], 0.5) 
set_pi(Pi, S[2], A[1], 0.5)   
set_pi(Pi, S[2], A[4], 0.5)   
set_pi(Pi, S[3], A[1], 0.5)  
set_pi(Pi, S[3], A[3], 0.5)   


print("----动作执行概率字典:----")
display_dict(Pi)

print("----价值函数字典:----")
V = {}
display_dict(V)


def compute_q(MDP, V, s, a):
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s,a,s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s,a) + gamma * q_sa
    return q_sa




def compute_v(MDP, V, Pi, s):
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s,a) * compute_q(MDP, V, s, a)
    return v_s




#根据当前策略使用回溯法来更新状态价值，本章不做要求
def update_V(MDP, V, Pi):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        #set_value(V_prime, s, V_S(MDP, V_prime, Pi, s))
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime


def policy_evaluate(MDP, V, Pi, n):
    for i in range(n):
        V = update_V(MDP, V, Pi)
        #display_dict(V)
    return V


# print(MDP[4])
V = policy_evaluate(MDP, V, Pi, 100)
display_dict(V)

v = compute_v(MDP, V, Pi, "第三节课")
print("第三节课在当前策略下的最终价值:{:.2f}".format(v))