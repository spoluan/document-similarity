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
        self.q_values = util.Counter() # 定義 Q-value 型態
        """
        上述寫法為用 Counter 初始化 Q-value
        舉例來說假設有一個 state 為 (0, 0)
        而他可以向北走
        那麼此 Q-value 會存在 self.q_values[((0, 0), "north")] 中
        """

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)] # 直接回傳 state 經由 action 的 Q-value

    def computeValueFromQValues(self, state): # 回傳 Q-value 最大的
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        """
        此 function 會回傳當前 state 中最大的 Q-value
        先找出當前 state 狀態下的 legal actions
        再將 state 與 legal actions 各自對應的 Q-value 用 buf 存起來
        如果 buf 為空的代表已經走到 terminal state, 所以回傳 0.0
        否則回傳 buf 中值最大的
        """
        legalActions = self.getLegalActions(state) # 取得 leagal actions

        buf = []
        for item in legalActions:
          buf.append(self.getQValue(state, item)) # 將 Q-value 依序加到 buf 中

        if len(buf) == 0: # 走到 terminal state 的情況下回傳 0.0
          return 0.0
        return max(buf) # 其他情況回傳所有 Q-value 的最大值

    def computeActionFromQValues(self, state): # 回傳最佳情況應該向哪移動
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        """
        此 function 會回傳在當前 state 中向哪移動會是最好的選擇
        首先一樣找出當前 state 狀態下的 legal actions
        如果沒有 legal actions, 代表走到 terminal state, 直接回傳 None
        接下來先取得當前 state 中 Q-value 的最大值, 並存在 maxValue 中
        只要在 legal actions 中的 Q-value 值和 maxValue 相等
        就將該 action 存在 best 中
        會有此做法是因為在一個 state 中各個方向可能會有相同的 Q-value
        所以要將同為 maxValue 的 action 都存起來, 之後再透過 random 隨機選
        """
        legalActions = self.getLegalActions(state) # 取得 leagal actions

        if len(legalActions) == 0: # 走到 terminal state 的情況下回傳 None
          return None
        
        maxValue = self.getValue(state) # 找出 Q-value 最大值
        best = []
        for item in legalActions:
          if self.getQValue(state, item) == maxValue: # 如果 Q-value 等於 maxValue, 將當前 action 加到 list 中
            best.append(item)

        return random.choice(best) # 從可以走的 best legal action 中隨機選一個

    def getAction(self, state): # 回傳在有 epsilon 的情況下應該怎麼走
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
        """
        此 function 會回傳有 epsilon 的情況下
        state 向哪移動會是最好的選擇
        如果 leagal action 為空, 直接回傳 None
        否則透過丟硬幣 ( random ) 的方式去決定 action
        如果小於 epsilon, 則 action 隨機從 legal action 中選一個
        否則就選當前的最佳路徑
        """
        if len(legalActions) == 0: # legal action 為空, 回傳 None
          return action

        if util.flipCoin(self.epsilon): # 小於 epsilon 所以隨機走
          action = random.choice(legalActions)
        else: # 大於等於 epsilon 所以選最佳路徑
          action = self.getPolicy(state)
        
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
        """
        透過 Q-learning 的公式計算 Q-value 並更新值
        """
        old = (1 - self.alpha) * self.getQValue(state, action)
        sample = reward + self.discount * self.getValue(nextState)
        new = self.alpha * sample
        self.q_values[(state, action)] = old + new

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
