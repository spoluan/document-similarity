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

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.q_values[(state, action)]  #回傳self在這一個state的QValue


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        legalActions = self.getLegalActions(state)  #取得self在這個state能夠進行的action
        if len(legalActions)==0:  #如果沒有可進行的action，return 0.0
          return 0.0

        tmp = util.Counter() #複製一個util.Counter()
        for action in legalActions:  #給定self在這個state的所有legalAction一個QValue
          tmp[action] = self.getQValue(state, action)

        return tmp[tmp.argMax()]  #回傳tmp中索引值最大的那個

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        actions = self.getLegalActions(state)  #取得self在這個state能夠進行的action
        best_action = None
        max_val = float('-inf')
        for action in actions:  #比較所有self在這個state的legalAction的QValue，回傳最大的那個action
          q_value = self.q_values[(state, action)]
          if max_val < q_value:
            max_val = q_value
            best_action = action
        return best_action

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
        #util.raiseNotDefined()
        explore = util.flipCoin(self.epsilon)
        if explore:
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        #擲一枚硬幣，如果是正面就隨機回傳一個legalAction，若是反面則回傳最優解
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
        #util.raiseNotDefined()
        old_q_value = self.getQValue(state, action)
        old_part = (1 - self.alpha) * old_q_value
        reward_part = self.alpha * reward
        #計算舊期望值和新期望值
        if not nextState:  #如果還沒決定好nextState，QValue維持原樣
          self.q_values[(state, action)] = old_part + reward_part
        else: #如果已經決定好nextState，則重新計算QValue
          nextState_part = self.alpha * self.discount * self.getValue(nextState)
          self.q_values[(state, action)] = old_part + reward_part + nextState_part

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
        #util.raiseNotDefined()
        features = self.featExtractor.getFeatures(state, action) #取得self在這個state的這個action的feature
        total = 0
        for i in features:  #計算QValue
            total += features[i] * self.weights[i]
        return total

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)  #計算加權值
        features = self.featExtractor.getFeatures(state, action)  #取得self在這個state的這個action的feature
        for i in features:  #更新self在每個feature的weight
            self.weights[i] = self.weights[i] + self.alpha * diff * features[i]

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print ("L : {0}".format(self.alpha))  #輸出self的learning rate
            print ("D : {0}".format(self.gamma))  #輸出self的Discounting rate
            print ("E : {0}".format(self.epsilon))  #輸出self的exploration prob
            print ("nT: {0}".format(self.numTraining))  #輸出self的numTraining
            print ("F :")
            for i in features:  #輸出self在每個feature的weight
              print ("{0} : {1}".format(i, self.weights[i]))
            pass
