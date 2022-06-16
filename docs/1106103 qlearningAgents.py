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
        # 給Q初始值
        self.QValue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # 如上所述return Q-value
        return self.QValue[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        Move = self.getLegalActions(state)
        values = []

        # 重複動作並加上該state的任何action之QValue
        Temp = util.Counter()
        if len(Move) != 0:
            for action in Move:
                Temp[action] = self.getQValue(state, action)
        # 不動，則回傳0.0
        else:
            return 0.0
        # 回傳最大值
        return Temp[Temp.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # 求最佳動作
        BestValue = self.getValue(state)
        BestChoice = [action for action in self.getLegalActions(state)
                      if self.getQValue(state, action) == BestValue]
        # 不動 回傳None
        if not len(BestChoice):
            return None
        # 回傳Q值最大的選擇
        else:
            return random.choice(BestChoice)

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
        # 以epsilon機率取random action
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        # 反之，則回傳best
        else:
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
        # Q - learning update alg. :
        # Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample
        # 求舊Q值
        LastQ = self.getQValue(state, action)
        # 求sample變數
        sample = reward + self.discount * \
            self.computeValueFromQValues(nextState)
        # perform the update and add it to our qVals dict-counter
        self.QValue[(state, action)] = (1 - self.alpha) * \
            LastQ + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)






    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # 用Identity extractor求feature
        features = self.featExtractor.getFeatures(state, action)
        # 求weights
        weights = self.getWeights()
        # 以features * weights 求Q值
        QValue = features * weights
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # diff值 = reward + gamma*Q(s', a') - Q(s,a)
        diff = reward + self.discount * \
            self.computeValueFromQValues(
                nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        # weight是空的, 則將其initialize為0
        if len(weights) == 0:
            weights[(state, action)] = 0
        features = self.featExtractor.getFeatures(state, action)
        # 更新係數矩陣
        for x in features:
            features[x] = features[x] * self.alpha * diff
        # 將weights與它們對應的新特徵值相加
        weights.__radd__(features)
        # update weights
        self.weights = weights.copy()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("Final weights vector: ")
            print(self.weights)
            pass
