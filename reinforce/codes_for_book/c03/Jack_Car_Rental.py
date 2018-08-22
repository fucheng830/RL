# -*- coding: utf-8 -*-

"""
租车问题
-----------------

State = (cars_loc1, cars_loc2) 每方最多20辆车

行为 每晚最多移动-5～5辆车从loc1到loc2

奖励 R 每租出去1辆车，奖励$10

状态转移概率 租车还车服从泊松分布

第一个地点 平均租车3，返还3
第二个地点 平均租车4， 返还2
"""

import numpy as np
from tqdm import tqdm

MAX_CARS = 20
MAX_MOVE = 5
S = []
for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        S.append((i,j))

A = [i for i in range(-MAX_MOVE,MAX_MOVE+1)]
params = {
    "loc1_rent":3,
    "loc1_return":3,
    "loc2_rent":4,
    "loc2_return":2
}

def dynamics(s, a):
    '''Jack Car 租赁问题的动力学，包含泊松分布等随机性
    Args:
        s 两地(loc1, loc2)汽车库存数 tuple (n1, n2) 0<=n1,n2<=20
        a 转场汽车数 int 定义为从 loc1转运a辆车至loc2地，负数表示反向搬运 -5<=a<=5
    Returns:
        s_prime 后续状态 tuple(int, int) (n1,n2)
        reward float 奖励，两地合算，每租出1辆车，奖励10
        is_end Bool 本例可循环，没有终止状态，返回False
    '''
    s_prime, reward, is_end = None, None, False
    # Add your code here
    return s_prime, reward, is_end

def poisson_prob(n, lamda):
    '''计算从一个参数为n定义的泊松分布中，采样输出为lamda的概率
    '''
    if lamda < 0:
        return 0.0
    return np.power(lamda,n)*np.power(np.e, -lamda) / np.math.factorial(n)

def _need_return(car_rent, loc_s0, loc_s1):
    '''借出car_rent辆车，需要有多少辆车返回才能满足从loc_s0->loc_s1
    '''
    return car_rent + (loc_s1 - loc_s0)

def prob_one_location(n_rent, n_return, loc_s0, loc_s1):
    '''计算由借出和返还泊松参数决定的随机事件中，某一地点的汽车数量
    从loc_s0变为loc_s1的概率。
    '''
    max_for_rent = loc_s0 # 当前最大可借出数量
    prob = 0 # 初始概率设为0
    for car_rent in range(0, max_for_rent):# 对于每一个借出的可能性
        prob_rent = poisson_prob(n_rent, car_rent)
        # 借出car_rent辆车，需要有car_return辆返回才能满足最终汽车数为loc_s1
        car_need_return = _need_return(car_rent, loc_s0, loc_s1)
        if car_need_return < 0: 
        # <0 说明借的不够多，将在借出里考虑计算，不重复计算
            prob_return = 0.0
            # continue
        prob_return = poisson_prob(n_return, car_need_return)
        prob += prob_rent * prob_return # 概率乘积
    return prob
    
def reward_one_location(n_rent, loc_aftmv):
    reward = 0
    for cars in range(loc_aftmv + 1):
        reward += cars * poisson_prob(n_rent, cars)
    return reward

# print _need_return(3,5,2)
# print prob_one_location(3,3,4,3)
# print reward_one_location(3,5)

def P(s, a, s1):
    '''重新改写状态转换概率
    '''
    n1, n2 = s # 当前1,2两地汽车数量数目
    if not(0<= n1 <= MAX_CARS and 0<= n2 <= MAX_CARS):
        #"初始状态不合法"
        return 0.0
    n1_aftmv, n2_aftmv = n1 - a, n2 + a # 移动a辆汽车后两地汽车数量
    if not(0<= n1_aftmv <= MAX_CARS and 0 <= n2_aftmv <= MAX_CARS): 
        # "转场的汽车数不符合要求"
        return 0.0
    n1_prime, n2_prime = s1
    prob = prob_one_location(params["loc1_rent"],
                             params["loc1_return"],
                             n1_aftmv, n1_prime)
    prob *= prob_one_location(params["loc2_rent"],
                              params["loc2_return"],
                              n2_aftmv, n2_prime)
    return prob


def R(s, a):
    n1, n2 = s # 当前1,2两地汽车数量数目
    if not(0<= n1 <= MAX_CARS and 0<= n2 <= MAX_CARS):
        #"初始状态不合法"
        return 0.0
    n1_aftmv, n2_aftmv = n1 - a, n2 + a # 移动a辆汽车后两地汽车数量
    if not(0<= n1_aftmv <= MAX_CARS and 0 <= n2_aftmv <= MAX_CARS): 
        # "转场的汽车数不符合要求"
        return 0.0
    reward = 0
    reward += reward_one_location(params["loc1_rent"], n1_aftmv)
    reward += reward_one_location(params["loc2_rent"], n2_aftmv)
    return reward
    
def set_value(V, s, v):
    loc1, loc2 = s
    V[loc1,loc2] = v
    
def get_value(V, s):
    loc1, loc2 = s
    return V[loc1, loc2]

def display_V(V):
    print(V)
    
gamma = 1
MDP = S, A, R, P, gamma   


'''
def P(s, a, s1):
    s_prime, _, _ = dynamics(s, a)
    return s1 == s_prime

def R(s, a):
    _, r, _ = dynamics(s, a)
    return r
'''
def get_prob(P, s, a, s1):
    return P(s, a, s1)

def get_reward(R, s, a):
    return R(s, a)

########################

def get_pi(Pi, s, a, MDP = None, V = None):
    return Pi(MDP, V, s, a)

def uniform_random_pi(MDP = None, V = None, s = None, a = None):
    _, A, _, _, _ = MDP
    n = len(A)
    return 0 if n == 0 else 1.0/n

def greedy_pi(MDP, V, s, a):
    S, A, P, R, gamma = MDP
    max_v, a_max_v = -float('inf'), []
    for a_opt in A:# 统计后续状态的最大价值以及到达到达该状态的行为（可能不止一个）
        s_prime, reward, _ = dynamics(s, a_opt)
        v_s_prime = get_value(V, s_prime)
        if v_s_prime > max_v:
            max_v = v_s_prime
            a_max_v = [a_opt]
        elif(v_s_prime == max_v):
            a_max_v.append(a_opt)
    n = len(a_max_v)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_v else 0.0


def epsilon_greedy_pi(MDP, V, s, a, epsilon = 0.1):
    if MDP is None:
        return 0.0
    _, A, _, _, _ = MDP
    m = len(A)
    greedy_p = greedy_pi(MDP, V, s, a)
    if greedy_p == 0:
        return epsilon / m
    # n = int(1.0/greedy_p)
    return (1 - epsilon + epsilon/m) * greedy_p


def compute_q(MDP, V, s, a):
    '''根据给定的MDP，价值函数V，计算状态行为对s,a的价值qsa
    '''
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s,a) + gamma * q_sa
    return q_sa


def compute_v(MDP, V, Pi, s):
    '''给定MDP下依据某一策略Pi和当前状态价值函数V计算某状态s的价值
    '''
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return v_s        

def update_V(MDP, V, Pi):
    '''给定一个MDP和一个策略，更新该策略下的价值函数V
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        #print("in update V, for state s")
        set_value(V_prime, s, compute_v(MDP, V_prime, Pi, s))
    return V_prime


def policy_evaluate(MDP, V, Pi, n):
    '''使用n次迭代计算来评估一个MDP在给定策略Pi下的状态价值，初始时价值为V
    '''
    for i in tqdm(range(n)):
        #print("====第{}次迭代====".format(i+1))
        V = update_V(MDP, V, Pi)
        #display_V(V)
    return V

def policy_iterate(MDP, V, Pi, n, m):
    cur_Pi = Pi
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi = epsilon_greedy_pi
        #print("改善了策略")
    return V

# 价值迭代得到最优状态价值过程
def compute_v_from_max_q(MDP, V, s):
    '''根据一个状态的下所有可能的行为价值中最大一个来确定当前状态价值
    '''
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa >= v_s:
            v_s = qsa
    return v_s

def update_V_without_pi(MDP, V):
    '''在不依赖策略的情况下直接通过后续状态的价值来更新状态价值
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
    return V_prime

def value_iterate(MDP, V, n):
    '''价值迭代
    '''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
        display_V(V)
    return V

V = np.zeros((MAX_CARS+1, MAX_CARS+1))
V_pi = value_iterate(MDP, V, 4)
V_pi = policy_evaluate(MDP, V, uniform_random_pi, 1)
# display_V(V_pi)
print V_pi

# import matplotlib.pyplot as plt
# plt.imshow(V_pi, cmap=plt.cm.cool, interpolation=None, origin="lower")#, extent=[0, 11, 0, 22])
# v_a = np.range((MAX_CARS+1,MAX_CARS+1))
# for i in range(MAX_CARS+1):
#     for j in range(MAX_CARS+1):
#         v_a[i,j] = greedy_policy