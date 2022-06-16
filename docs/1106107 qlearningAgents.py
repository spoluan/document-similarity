# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

# Q1,Q2,Q3僅跑這邊
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
        返回該狀態合適的行為資料
          which returns legal actions for a state
    """
    def __init__(self, **args):        
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # 用uitl.counter作為搜尋值的索引，後續getQValue會需要取得以欄位值(狀態位置,動作北西南東)=>value
        self.q_Value = util.Counter()
        
    # 只能用此函數調用Q值
    def getQValue(self, state, action):
        """
          回復 (狀態,動作)
          如果沒有看過的狀態應返回0,0
          否則回復Q節點值          
        """
        "*** YOUR CODE HERE ***"
        # 回復取的各狀態欄位之值(state, action)為其索引index(key value)，並將北西南東之VALUE帶出        
        return self.q_Value[(state, action)]

    def computeValueFromQValues(self, state):
        """
          回復在這個狀態下最大動作的 Q(state,action)
          Returns max_action Q(state,action)
        """
        "*** YOUR CODE HERE ***"
        # 帶出目前所在節點位置可執行之方位
        # state=座標，若為TERMINAL_STATE則為出口點，若為()空陣列另外做處理
        action_List = self.getLegalActions(state)    
        # 將各方位可帶出之值存入
        return_List = []
        # 判斷如果為空陣列則回覆0
        if len(action_List) == 0:            
            return 0
        else:
          # 將各方位取出，並將目前位置所帶出之各方位值存入return_List中
            for action_List_Value in action_List:              
                return_List.append(self.getQValue(state, action_List_Value))
        # 取最大值        
        return max(return_List)

    def computeActionFromQValues(self, state):
        """
        計算狀態位置最佳動作，如果沒有合理得動作，在終點狀態下就是這樣要回復none
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 帶出目前所在節點位置可執行之方位
        # state=座標，若為TERMINAL_STATE則為出口點，若為()空陣列另外做處理
        action_List = self.getLegalActions(state)
        # 全部的動作
        all_Actions = []
        
        # 先判斷是否為空陣列()，若是依此函示提示為terminal處回覆none，若否再將全部得動作做值判斷
        if len(action_List) == 0:
            return None
        #iterate over actions and append all the qvalues and actions to 'allActions'
        else:
            for action_List_Value in action_List:
              # 將各方位可得之值帶出              
                all_Actions.append((self.getQValue(state, action_List_Value), action_List_Value))
            # 取出最大值回復執行之動作
        return max(all_Actions)[1]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        # 帶出目前所在節點位置可執行之方位
        # state=座標，若為TERMINAL_STATE則為出口點，若為()空陣列另外做處理
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # With probability self.epsilon
        probability = self.epsilon
        # with probability self.epsilon, we should take a random action and take the best policy action otherwise.
        # flipCoin 是用在判斷傳入機率值是否比隨機值大，若是 true 否則為false
        if util.flipCoin(probability):
          # random action
            return random.choice(legalActions)        
        else:
          # 選擇最好最好得動作值回復
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # 先將提示之action 之Q VALUE 取出
        Qvalue = self.getQValue(state, action)
        #套用公式        
        #perform the update and add it to our qVals dict-counter
        # q_Value= (1-學習率)*Qvalue(舊的位置的值) +學習率*(獎勵值+折扣*下一個位置最大的值)
        self.q_Value[(state, action)] = (1-self.alpha)*Qvalue + self.alpha*(reward + self.discount*self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

# 吃豆人
class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        # 從QLearning處帶出getAction做處理，可得知getAction會傳回動作
        action = QLearningAgent.getAction(self,state)
        # print(action)
        self.doAction(state,action)
        return action

# 近似Q
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
