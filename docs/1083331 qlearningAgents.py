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

        # 初始化存qvalue的容器
        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        #如果state不在容器裡就回傳0.0
        if not self.qvalues[(state, action)]:
            return 0.0

        #如果在容器裡就會傳相對應的值
        else:
            return self.qvalues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        # 因為要計算value，value是在該state能取得的最大值
        # 所以遍歷qvalue，找到最大的qvalue作為該state的value值
        
        # 先取得所有合法的action
        all_action = self.getLegalActions(state)

        #如果沒有任何一個合法的動作就回傳0.0
        if len(all_action) == 0:
            return 0.0

        # 如果有合法的動作則遍歷所有的qvalue算該state的值
        else:
            # 先將第一個合法action的值假設為最大的值
            max_value = self.getQValue(state, all_action[0])

            # 遍歷所有的合法的action
            for i in range(0,len(all_action)):
                
                value = self.getQValue(state, all_action[i])
                # 若是比之前所有的value還大則將現在的value設為最大的value
                if value > max_value:
                    max_value = value
            # 最後回傳最大值
            return max_value


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # 採取最好的action就是找value和qvalue相等的那個action
        # 因此先找出蓋state德value後
        # 再找找看哪個action的qvalue和value相等就回傳該action

        #先取得該state的最大value
        value = self.getValue(state)
        # 再取得所有合法的action
        all_action = self.getLegalActions(state)
        # 遍歷所有合法德action 
        for action in all_action:
            # 若是該action的value和最大的value相等則回傳該action
            if self.getQValue(state, action) == value:
                return action
    
        return None 


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

        # 取得action時因為有部分機率不會按照指示走，故先看是否要random
        # 若是不用random，則回傳qvalue最大的合法action

        # 如果沒有任何一個合法的動作就回傳None
        if len(self.getLegalActions(state)) == 0:
            return None
        
        # 使用filpCoin函式來決定是否要Random
        random_ = util.flipCoin(self.epsilon)

        # 如果要random則隨機回傳一個action
        if random_:
            return random.choice(legalActions)

        # 如果不要random 則回傳最大value的action
        else:
            # 先預設最大的value為第一個合法的action
            best_qvalue = self.getQValue(state, legalActions[0])
            i = 0
            pos = 0

            for i in range(0, len(legalActions)):
                qvalue = self.getQValue(state, legalActions[i])
                # 若遇到更大的value就記住該位置
                if qvalue > best_qvalue:
                    best_qvalue = qvalue
                    pos = i
            # 最後回傳該位置的action
            return legalActions[pos] 

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # 取得要計算的所有參數(discount, nextvalue, alpha等)
        alpha = self.alpha
        gamma = self.discount
        nextvalue = self.getValue(nextState)
        qvalue = self.getQValue(state, action)

        # 計算新的value(套公式)
        newvalue = (1 - alpha) * qvalue + alpha * (reward + gamma * nextvalue)

        # 最後更新value
        self.qvalues[(state, action)] = newvalue 

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
