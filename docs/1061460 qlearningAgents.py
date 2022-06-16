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
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        #使用dictionary的資料結構做資料查詢
        self.Q={}
        #原因：以dictionary做可以直接使用(state,action)當作key值存取Qvalue。

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) in self.Q: #當(state,action)在dictionary裡
            return self.Q[(state,action)] #回傳其Qvalue
        else: #否則回傳0.0
            return 0.0
        #原因：當computeValueFromQValues與computeActionFromQValues要access qvalue時，只能透過呼叫getQValue。
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #設一個名為tmp_s_a的list儲存當前state合法action的Qvalue
        tmp_s_a = [] 
        if not self.getLegalActions(state): #檢查這個state是否有合法的action
            return 0.0 #沒有就回傳0.0
        else: #合法就將action透過for迴圈將getQValue(state, action)append進tmp_s_a
            for action in self.getLegalActions(state):
                tmp_s_a.append(self.getQValue(state, action))
        return max(tmp_s_a) #回傳max_action Q(state,action)
        #原因：透過實作computeValueFromQValues，才能在update計算nextstate的Qvalue時呼叫這個method，透過給state的值來找到當前最佳的qvalue。

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #設一個名為tmp_s_a的list儲存當前state合法action的value和此action這對pair
        tmp_s_a = [] 
        if not self.getLegalActions(state):#檢查這個state是否有合法的action
            return None #沒有就回傳none
        else: #合法就把(qvalue,action)的pair append進tmp_s_a
            for action in self.getLegalActions(state):
                tmp_s_a.append((self.getQValue(state, action), action))
            for i in tmp_s_a: #設一個ans存max(tmp_s_a)
              if i == max(tmp_s_a):
                  ans=i       
        return ans[1] #回傳best action
        #原因：實作這個computeActionFromQValues才能在getAction呼叫這個method時，透過給state的值來找到當前最佳的action。

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
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #實作epsilon-greedy action selection
        if util.flipCoin(self.epsilon): #當機率為self.epsilon時
            action = random.choice(legalActions) #從getLegalActions(state)裡透過random.choice(list)來random選出action
        else: #當機率不為self.epsilon時
            action = self.computeActionFromQValues(state) #則透過computeActionFromQValues(state)回傳 best action
        #原因：實作getAction method才能在機率為epsilo時隨機選出action。
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #先運用computeValueFromQValues計算nextState的Qvalue
        next_v = self.computeValueFromQValues(nextState)
        #再運用qlearning的更新公式計算新的Qvalue
        next_v = (1-self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * next_v)
        #最後運用dictionary的方式放入(state,action)相對應的value裡
        self.Q[(state, action)] = next_v
        #原因：實作update method才能將每個action帶來的reward更新到這個Q-table，且dictionary在一開始沒東西時也能初始化值。

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


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
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

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
