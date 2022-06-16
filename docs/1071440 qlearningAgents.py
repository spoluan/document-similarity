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
        self.q_values = util.Counter()
        # 初始化 Q-values，用util.Counter()保存每個狀態的Q值

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q_values[(state, action)] #回傳現在q_values的狀態跟動作
        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

       
        if len(self.getLegalActions(state))==0: #沒有legal actions(terminal state)，回傳 0.0
          return 0.0
        temp = util.Counter()
        for action in self.getLegalActions(state):
            temp[action] = self.getQValue(state, action)
        # 把每個action,state取出來
        return temp[temp.argMax()] #回傳當中的最大值

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        
        actions = self.getLegalActions(state)
        all_Actions = []
        
        if len(actions) == 0: #沒有合法actions，回傳none
            return None  
        else:     #跑過所有action，把qvalues跟actions存在allActions[]裡
            for action in actions:
                all_Actions.append((self.getQValue(state, action), action))
            bestActions = [pair for pair in all_Actions if pair == max(all_Actions)] #取出最佳(最大值)qvalue,action組合
            bestActionPair = random.choice(bestActions) #如果最佳解不只一個，則隨機選一個qvalue,action組合
        #return the best action
        return bestActionPair[1]
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

        #選一個隨機的action
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else: #或回傳最佳解action
            return self.computeActionFromQValues(state)
        util.raiseNotDefined()
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
        #### Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample
        #### sample = R(s,a,s') + gamma*max(Q(s',a')) 
        #帶入公式計算 
        pre_Q = self.getQValue(state, action)
        sample = reward + self.discount*self.computeValueFromQValues(nextState)
        self.q_values[(state, action)] = (1-self.alpha)*pre_Q + self.alpha*sample
        # util.raiseNotDefined()


        
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
        features = self.featExtractor.getFeatures(state,action)
        #權重
        weights = self.getWeights()
        #做矩陣乘法，得最後Q
        total = 0
        for i in features:
            total += features[i] * self.weights[i]
            # print("test:",i)
        return total


        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        

        #####differece = reward + gamma*Q(s', a') - Q(s,a) 根據公式計算diff，更新權重矩陣
        diff = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        #如果權重矩陣為空-->初始化為0
        if len(weights) == 0:
            weights[(state,action)] = 0
        features = self.featExtractor.getFeatures(state, action)
        #  把每個features中 與aplpha跟diff相乘
        for key in features.keys():
            features[key] = features[key]*self.alpha*diff
        #加總權重和向量特徵
        weights.__radd__(features)
        #update weights
        self.weights = weights.copy()



        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # print("Approximate Q-Learning~~~~~~")
            # print("Learning rate(alpha) : {0}".format(self.alpha))
            # print("Exploration rate(epsilon) : {0}".format(self.epsilon))
            # print("Training episodes : {0}".format(self.numTraining))
          
            
            pass