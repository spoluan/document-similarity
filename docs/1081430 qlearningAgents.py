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
    def __init__(self, **args):#鎖
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qVals = util.Counter()

    def getQValue(self, state, action):#鎖
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #get the value corresponding to Q(state, action)
        return self.qVals[(state, action)]

        #util.raiseNotDefined()

    def computeValueFromQValues(self, state):#所求
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # no actions =>return 0
        nums = []
        if len(self.getLegalActions(state)) == 0:
            return 0
        #run 這個 state 所有的 qvalue action
        else:
            for move in self.getLegalActions(state):
                nums.append(self.getQValue(state, move))
        #都放進nums後回傳最大值
        return max(nums)
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):#所求
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)

        if not legalActions:#legal action不存在的話 return none 結束(篩選用
            return None
        else:
            # 確認有沒有比bestaction更高的數字 如果有就換
            bestAction = 'none', -288741
            for move in legalActions:
                qvalue = self.getQValue(state, move)
                if qvalue > bestAction[1]:
                    bestAction = move, qvalue
            return bestAction[0]#一定要記得[0] de超久

    def getAction(self, state):#所求
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
        legal_actions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #利用epsilon可能地選到隨機的action
        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)#random funtion
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):#索求
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #sample從reward ,discount self.Qvalue取得
        sample = reward + self.discount*self.computeValueFromQValues(nextState)
        #加到qvals的 dict counter
        self.qVals[(state, action)] = (1-self.alpha)*self.getQValue(state, action) + self.alpha*sample
        

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

    def getQValue(self, state, action):#所求
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # 先從 featExtractor 取出 current state 還有 action
        features = self.featExtractor.getFeatures(state, action)
        
        # 先初始化較好
        All_FeaturesWeight = 0
        
        # run所有features 也加所有 weighted feature 給 total allfeature
        for move in features:
            All_FeaturesWeight += self.weights[move] * features[move]
        
        # 回傳weighted features的總和
        return All_FeaturesWeight

    def update(self, state, action, nextState, reward):#所求
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)
        
        #new_value = qvalue + alpha * (reward + disc * next_value - qvalue)
        new_value = (1-self.alpha) * qvalue + self.alpha * (reward + self.discount * next_value)
        
        self.setQValue(state, action, new_value) 

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
