# -*- coding: utf-8 -*-

"""
21点游戏环境
-----------------

Description of this Module.
"""
from __future__ import print_function
from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
from copy import deepcopy

import numpy as np

from utils import str_key, set_dict, get_dict
import pdb

class Arena(object):
    '''负责游戏管理
    '''
    def __init__(self, display = None, A = None):
        self.cards = ['A','2','3','4','5','6','7','8','9','10','J','Q',"K"]*4
        self.card_q = Queue(maxsize = 52) # 洗好的牌
        self.cards_in_pool = [] # 已经用过的公开的牌  
        self.display = display
        self.episodes = [] # 产生的对局信息列表
        self.load_cards(self.cards)# 把初始状态的52张牌装入发牌器
        self.A = A # 获得行为空间

    def load_cards(self, cards):
        '''把收集的牌洗一洗，重新装到发牌器中
        Args:
            cards 要装入发牌器的多张牌 list
        Return:
            None
        '''
        _cards = deepcopy(cards)
        shuffle(_cards) # 洗牌
        for card in _cards:# deque数据结构只能一个一个添加
            self.card_q.put(card)
        return
       
    def reward_of(self, dealer, player):
        '''判断玩家奖励值，附带玩家、庄家的牌点信息
        '''
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, player_points, dealer_points, useable_ace
    
    def serve_card_to(self, player, n = 1):
        '''给庄家或玩家发牌，如果牌不够则将公开牌池的牌洗一洗重新发牌
        Args:
            player 一个庄家或玩家 
            n 一次连续发牌的数量
        Return:
            None
        '''
        cards = []  #将要发出的牌
        for _ in range(n):
            # 要考虑发牌器没有牌的情况
            if self.card_q.empty():
                self._info("\n发牌器没牌了，整理废牌，重新洗牌;")
                shuffle(self.cards_in_pool)
                self._info("一共整理了{}张已用牌，重新放入发牌器\n".format(len(self.cards_in_pool)))
                assert(len(self.cards_in_pool) > 20) # 确保有足够的牌，将该数值设置成40左右时，如果玩家
                # 即使爆点了也持续的叫牌，会导致玩家手中牌变多而发牌器和已使用的牌都很少，需避免这种情况。
                self.load_cards(self.cards_in_pool) # 将收集来的用过的牌洗好送入发牌器重新使用
            cards.append(self.card_q.get()) # 从发牌器发出一章牌
        self._info("发了{}张牌({})给{}{};".format(n, cards, player.role, player))
        #self._info(msg)
        player.receive(cards) # 牌已发给某一玩家
        player.cards_info()

        
    def _info(self, message):
        if self.display:
            print(message, end="")
        
    def recycle_cards(self, *players):
        '''回收玩家手中的牌到公开使用过的牌池中
        '''
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards() # 玩家手中不再留有这些牌
                
    def play_game(self, dealer, player):
        '''玩一局21点，生成一个状态序列以及最终奖励（中间奖励为0）
        Args：
            dealer/player 庄家和玩家手中的牌 list
        Returns:
            tuple：episode, reward
        '''
        #self.collect_player_cards()
        self._info("========= 开始新一局 =========\n")
        self.serve_card_to(player, n=2) # 发两张牌给玩家
        self.serve_card_to(dealer, n=2) # 发两张牌给庄家
        episode = [] # 记录一个对局信息
        if player.policy is None:
            self._info("玩家需要一个策略")
            return
        if dealer.policy is None:
            self._info("庄家需要一个策略")
            return
        while True:
            action = player.policy(dealer)
            # 玩家的策略产生一个行为
            self._info("{}{}选择:{};".format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action)) # 记录一个(s,a)
            if action == self.A[0]: # 继续叫牌
                self.serve_card_to(player) # 发一张牌给玩家
            else: # 停止叫牌
                break
        # 玩家停止叫牌后要计算下玩家手中的点数，玩家如果爆了，庄家就不用继续了        
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        
        if player_points > 21:
            self._info("玩家爆点{}输了，得分:{}\n".format(player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward)) # 预测的时候需要形成episode list后同一学习V
            # 在蒙特卡洛控制的时候，可以不需要episodes list,生成一个episode学习一个，下同
            self._info("========= 本局结束 ==========\n")
            return episode, reward
        # 玩家并没有超过21点
        self._info("\n")
        while True:
            action = dealer.policy() # 庄家从其策略中获取一个行为
            self._info("{}{}选择:{};".format(dealer.role, dealer, action))
            if action == self.A[0]: # 庄家"继续要牌":
                self.serve_card_to(dealer)
                # 停止要牌是针对玩家来说的，episode不记录庄家动作
                # 在状态只记录庄家第一章牌信息时，可不重复记录(s,a)，因为此时玩家不再叫牌，(s,a)均相同
                # episode.append((get_state_name(dealer, player), self.A[1]))
            else:
                break
        # 双方均停止叫牌了    
        self._info("\n双方均了停止叫牌;\n")
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        player.cards_info() 
        dealer.cards_info()
        if reward == +1:
            self._info("玩家赢了!")
        elif reward == -1:
            self._info("玩家输了!")
        else:
            self._info("双方和局!")
        self._info("玩家{}点,庄家{}点\n".format(player_points, dealer_points))
        
        self._info("========= 本局结束 ==========\n")
        self.recycle_cards(player, dealer) # 回收玩家和庄家手中的牌至公开牌池
        self.episodes.append((episode, reward)) # 将刚才产生的完整对局添加值状态序列列表，蒙特卡洛控制不需要
        return episode, reward
    
    def play_games(self, dealer, player, num=2, show_statistic = True):
        '''一次性玩多局游戏
        '''
        results = [0, 0, 0]# 玩家负、和、胜局数
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode ,reward)
        if show_statistic:
            print("共玩了{}局，玩家赢{}局，和{}局，输{}局，胜率：{:.2f},不输率:{:.2f}"\
              .format(num, results[2],results[1],results[0],results[2]/num,(results[2]+results[1])/num))
        pass
    
if __name__ == '__main__':
    env = Arena()
    env.load_cards(env.cards)
    