# -*- coding: utf-8 -*-

"""
迭代法策略评估Iterative Policy Evaluation
-----------------

实验迭代法策略评估 https://zhuanlan.zhihu.com/p/28084955.
"""
import numpy as np

class Env(object):
    
    def __init__(self):
        self.states
        self.agent_cur_state 
        self.observation_space
        self.action_space
        
    def run(self):
        pass

    def _p(self, lam):
        return np.random.poisson(lam=lam, size=100000)
    
    
    
class Agent(object):
    
    def __init__(self):
        self.env = Env()
        self.Q = {}
        self.state = None
        self.reward = None
    
    def performPolicy(self, state):
        pass
    
    def act(self, a):       # 执行一个行为
        return self.env.step(a)
    
    def learning(self): 
        pass
        
if __name__ == '__main__':
    e = Env()
    print e._p(2)
    