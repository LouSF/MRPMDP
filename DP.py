"""
1. 小型方格世界MDP建模
2. 策略评估
3. 价值迭代
"""


"""
1. 小型方格世界MDP建模：
"""

# 创建状态、动作空间
S = [i for i in range(16)]# 状态空间
A=["n","e","s","w"] # 动作空间 (n:北  e:东  s:南  w:西)
ds_actions={"n":-4,"e":1,"s":4,"w":-1}

# 模拟小型方格世界的环境动力学特征
def dynamics(s,a):# 环境动力学
    '''
    Args:
    s 当前状态 int 0-15
    a 行为 str in['n','e','s','w'] 分别表示北、东、南、西

    Returns: tuple (s_prime, reward, is_end)
    s_prime 后续状态
    reward 奖励值
    is_end 是否进入终止状态
    '''
    s_prime = s
    if (s % 4 == 0 and a == "w") or (s < 4 and a == "n") or ((s + 1) % 4 == 0 and a == "e") or (s > 11 and a == "s") or s in[0, 15]:
        pass
    else:
        ds = ds_actions[a]
        s_prime = s + ds
    reward = 0 if s in [0, 15] else -1
    is_end = True if s in [0, 15] else False
    return s_prime, reward, is_end

# P,R,将由dynamics动态生成
def P(s,a,s1):# 状态转移概率函数
    s_prime, _, _ = dynamics(s, a)
    #print(s1 == s_prime)
    return s1 == s_prime

def R(s,a):# 奖励函数
    _, r, _ = dynamics(s, a)
    #print(r)
    return r

# 建立MDP和策略
gamma = 1.00
MDP = S, A, R, P, gamma
def uniform_random_pi(MDP = None, V = None, s = None, a = None): # 均一随机策略
    _, A, _, _, _ = MDP
    n = len(A)
    return 0 if n == 0 else 1.0/n

def greedy_pi(MDP, V, s, a): # 贪婪策略
    """
    迭代每个动作，判断哪个动作产生的价值最大，并将它记录下来加入最大动作集
    判断输入的动作是否在最大动作集合中，在的话返回最大动作集大小的倒数，不在的话返回0
    之所以将随机策略和贪婪策略都设置为概率的形式是为了后面方便于价值的计算
    """
    S, A, P, R, gamma = MDP
    max_v, a_max_v = -float('inf'), []
    for a_opt in A:  # 统计后续状态的最大价值以及到达该状态的行为(可能不止一个)
        s_prime, reward, _ = dynamics(s, a_opt)
        v_s_prime = get_value(V, s_prime)
        if v_s_prime > max_v:
            max_v = v_s_prime
            a_max_v = [a_opt]
        elif (v_s_prime == max_v):
            a_max_v.append(a_opt)

    n = len(a_max_v)
    if n == 0: return 0.0
    return 1.0 / n if a in a_max_v else 0.0

def get_pi(Pi, s, a, MDP=None, V=None):
    return Pi(MDP, V, s, a)


# 辅助函数
def get_prob (P, s, a, s1):  # 获取状态转移概率
    #print(P(s, a, s1))
    return P(s, a, s1)

def get_reward(R, s, a):  # 获取奖励值
    #print(R(s, a))
    return R(s, a)

def set_value(V,s,v):# 设置价值字典
    V[s] = v

def get_value(V,s):# 获取状态价值
    #print(V[s])
    return V[s]

def display_V(V):# 显示状态价值
    for i in range(16):
        print('{0:>6.2f}'.format(V[i]),end = " ")
        if (i+1) % 4 == 0:
            print("")
    print()


"""
2. 策略评估：
"""
def compute_q(MDP, V, s, a):
    '''
    根据给定的MDP,价值函数V,计算状态行为对s,a的价值q_sa
    :param MDP: 马尔科夫决策函数
    :param V: 价值函数V
    :param s: 状态
    :param a: 行为
    :return: 在s状态下a行为的价值q_sa
    '''
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa

def compute_v(MDP, V, Pi, s):
    '''
    给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
    :param MDP: 马尔科夫决策过程
    :param V: 状态价值函数V
    :param Pi: 策略
    :param s: 某状态s
    :return: 某状态的价值v_s
    '''
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)#一个状态的价值可以由该状态下所有行为价值表达
    return v_s

def update_V(MDP, V, Pi):
    '''
    给定一个MDP和一个策略,更新该策略下的价值函数
    '''
    S, _, _, _,_ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v(MDP, V_prime, Pi, s))
    return V_prime

def policy_evaluate(MDP, V, Pi, n):
    '''
    使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值,初始时为V
    '''
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V


#测试随机策略
V = [0 for _ in range(16)]  # 状态价值
V_pi = policy_evaluate(MDP, V, uniform_random_pi, 1000)
print("策略评估: 使用随机策略")
display_V(V_pi)
#测试贪婪策略
V = [0 for _ in range(16)]  # 状态价值
V_pi = policy_evaluate(MDP, V, greedy_pi, 100)
print("策略评估: 使用贪婪策略")
display_V(V_pi)

"""
3. 策略迭代：
"""
def policy_iterate(MDP, V, Pi, n, m):
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi= greedy_pi
    return V

V=[0 for _ in range(16)]# 重置状态价值
V_pi = policy_iterate(MDP, V, greedy_pi, 1, 10)
print("策略迭代:")
display_V(V_pi)



"""
4. 价值迭代：
"""
# 价值迭代得到最优状态价值过程
def compute_v_from_max_q(MDP, V, s):
    '''根据一个状态的下所有可能的行为价值中最大一个来确定当前状态价值 '''
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa >= v_s:
            v_s = qsa
    return v_s

def update_V_without_pi(MDP, V):
    '''在不依赖策略的情况下直接通过后续状态的价值来更新状态价值 '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
    return V_prime

def value_iterate(MDP, V, n):
    '''价值迭代'''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


V_star = value_iterate(MDP, V, 10)
print("价值迭代:")
display_V(V_star)


# V=[0 for _ in range(16)]# 重置状态价值
# V_pi = policy_iterate(MDP, V, greedy_pi, 1, 1)
# display_V(V_pi)