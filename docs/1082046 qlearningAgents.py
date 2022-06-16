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


#from collections import defaultdict
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
        #初始化Q-value
        self.Q_value=util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #回傳指定位置及方向的Q_value值
        return self.Q_value[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #獲得所有合法的action
        legal_actions = self.getLegalActions(state)
        #如果list中沒有合法action，回傳0.0
        if not legal_actions:
            return 0.0
        #回傳所有合法action中的最大值
        temp=util.Counter()
        count=0
        max=-100000000
        for action in legal_actions:   
            if count==0:
               temp[action]=self.getQValue(state,action)
               max=temp[action]
            else:
                temp[action]=self.getQValue(state,action)
            count=count+1
            if temp[action]>max:
                max=temp[action]            
        return max           

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #獲得所有合法的action
        legal_actions = self.getLegalActions(state)
        #如果list中沒有合法action，回傳none
        if not legal_actions:
            return None
        #回傳Q-value中值最大所對應action
        max=-100000000
        maxaction=legal_actions[0]
        for action in legal_actions:
            qvalue=self.Q_value[(state,action)]
            if qvalue>max:
                max=qvalue
                maxaction=action
        return maxaction

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
        "*** YOUR CODE HERE ***" 
        #計算目前狀態下所要採取的行動(具隨機性的探索)，如果random<self.spsilon->隨機選一條路，否則走最佳路徑
        legal_actions = self.getLegalActions(state)      
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #更新Q-value 公式:Q(s,a)=(1-alpha)*Q(s,a)+alpha*sample 
        #sample=R(s,a,s')+discount*max(Q(s',a'))表新獲得的Q-value
        origin=self.getQValue(state,action)#先得到舊的Q-value
        if nextState:#如果nextState上有值，加sample
            sample = reward + self.discount * self.getValue(nextState)
            self.Q_value[(state, action)] = (1 - self.alpha) * origin + self.alpha * sample
        else:#如果nextState上無值，只要加reward
            self.Q_value[(state, action)] = (1 - self.alpha) * origin + self.alpha * reward

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
        #Approximate Q-Learing Q-value算法: Q(s,a)=w1*f1+w2*f2+w3*f3+......
        features = self.featExtractor.getFeatures(state, action)#將所有特徵值取出
        y=0
        for f in features:
            y=y+features[f]*self.weights[f]
        return y

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #更新wight值 weight=weight+alpha*difference*f(s,a)
        #difference > 0-> weight提高(特徵重要)
        #difference < 0-> weight降低(特徵不重要)
        features = self.featExtractor.getFeatures(state, action)#將所有特徵值取出
        sample = reward + self.discount * self.getValue(nextState)#計算新的Q-value
        difference = sample - self.getQValue(state, action)#difference為新的Q-value減舊的Q-value
        for f in features:#更新每個特徵的weight
            self.weights[f] = self.weights[f] + self.alpha * difference * features[f]

        
    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
