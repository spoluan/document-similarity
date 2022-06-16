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
        self.qValues = util.Counter()
        
        "*** YOUR CODE HERE ***"
        #會初始化 self.qValues(qtable) 的原因是我發現在
        #textGridworldDisplay.py中的displayQVlues()是
        #藉由我們寫的getQValue(state, action)更新視窗的，
        #所以只要在QLearningAgent裡建構可以存放qValues的
        #qtable就可以使用update(self, state, action, nextState, reward)
        #更新自己所建構的qtable，最後視窗畫面也會完成更新。

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        
        return self.qValues[(state, action)]
        util.raiseNotDefined()
        #呼叫getQValue(state, action)可以得知
        #目前這個state和action在qtable上的值，
        #可以用於更新qvalue和視窗畫面，有些function也會呼叫用來計算。

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        a = self.getLegalActions(state)
        if len(a) == 0:
          return 0.0

        action = self.getPolicy(state)

        return self.getQValue(state, action)

        util.raiseNotDefined()
        #computeValueFromQValues(self, state)可以用來計算目前
        #這個state最大的Qvalue，可以用於update(self, state, action, nextState, reward)
        #，計算下一個state的最大qValue。

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        a = self.getLegalActions(state)
        if len(a) == 0:
          return None
        else:
          max1 = float('-inf')
          actions = []
          actions_value = {}
          for i in a:
            actions_value[i] = self.getQValue(state,i)
            if actions_value[i] > max1:
              max1 = self.getQValue(state,i)

          for i in a:
            if actions_value[i] == max1:
              actions.append(i)

          return random.choice(actions)

        #computeActionFromQValues(self, state)是用來
        #計算state中最大的qvalue的action當作下一步要走的方向，
        #但由於可能會有qvalue一樣大的情況，所以可以用random.choice()進行隨機挑選最佳action。
        #不過在挑選動作之前，需要檢查這個state的那些action是合法的，如果都不合法就回傳None。
          
        util.raiseNotDefined()

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

        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        else:
          return self.computeActionFromQValues(state)
        
        
        return action
        util.raiseNotDefined()

        #藉由epsilon的方式給一定概率是否要嘗試走不是目前最佳action的step，
        #這樣有機會找到更好的policy，相反的也有一定的概率選擇最佳action。
        #可以調整epsilon的大小讓random的概率變小，慢慢收斂。

        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qValues[(state, action)] =  (1 - self.alpha)*self.getQValue(state,action) + self.alpha*(reward + self.discount*self.getValue(nextState))
        return self.qValues[(state, action)]
        util.raiseNotDefined()
        #選擇完action就是要update qtable，
        #藉由Alpha讓更新qValue時，可以讓之前所更新的舊qValue的占比更小，
        #將過去的影響隨時間逐漸變小，而此次更新的qValue影響較大。另外，reward是每
        #走一步所得到獎勵，discount則是讓整體policy可以更快得到更好的reward，
        #如果越後面才得到的reward會被discount，並且有助於收斂。

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
