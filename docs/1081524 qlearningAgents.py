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
from collections import defaultdict #add

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
        # defaultdict來初始化數組
        self.Q = defaultdict(lambda: defaultdict(float))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        """註解
        state代表不同格子/action代表一個格子裡的四個方向north,south,west,east
        state-->(0,0) /<class 'tuple'> ; action-->north /<class 'str'>
        self.Q[state] --> defaultdict(<class 'float'>, {'north': 0.0}) /<class 'collections.defaultdict'>
        or self.Q[state] --> defaultdict(<class 'float'>, {'north': 0.0, 'west': 0.0, 'south': 0.0})
        self.Q[state][action] --> 0.0 /<class 'float'>
        """
        return self.Q[state][action]#否則return the Q node value

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions=self.getLegalActions(state)
        if not legal_actions: #如果沒有legal action,代表case是在terminal state,return 0.0
            return 0.0
        else:
            max_Value=-1000000.0
            for i in legal_actions:
                max_Value=max(max_Value,self.getQValue(state, i))#在legal_actions中挑出max_Value
            return max_Value
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if not legal_actions:#如果沒有legal action,代表case是在terminal state,return None
            return None
        else:
            ValueFromQValues=self.computeValueFromQValues(state) #computeValueFromQValues回傳的是legal_actions中最大值的value
            best_action=[]
            for i in legal_actions:
                if self.getQValue(state,i)==ValueFromQValues:
                    best_action.append(i)#若是有多個action在best_action中，代表他們的目前的優先權是一樣的
            choice=random.randrange(0, len(best_action))#random.randrange(0,len(best_action)) --> 0~len(best_action)-1
            return best_action[choice] #在複數個best_action中隨機挑一個
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
        """註解
        util.flipCoin(0.3) -->0.3機率是True,0.7是False
        random.choice(list) -->list中隨機選擇
        如果util.flipCoin是True代表randonaction(外出探險)
        如果util.flipCoin是False代表take the best policy action
        """
        if not legalActions: #如果沒有legalActions,return None
            return None
        random_action=util.flipCoin(self.epsilon)#機率判斷是否要隨機action
        if random_action==True:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action
        util.raiseNotDefined()



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # gamma --> discount factor
        # Q-learning公式 --> Q*(s,a) = Q(s,a) + alpha * (reward + discount * max(Q(nexts,nexta)) - Q(s,a) )
        #帶入Q-learning公式去更新q-state
        #alpha * (reward + discount * max(Q(nexts,nexta)) - Q(s,a) )
        calculate = self.alpha * ( reward + (self.discount * self.computeValueFromQValues(nextState)) - self.Q[state][action])
        # Q*(s,a) = Q(s,a) + calculate
        Q = self.Q
        Q[state][action]=Q[state][action]+calculate
        #util.raiseNotDefined()

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
