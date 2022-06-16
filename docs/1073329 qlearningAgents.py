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
        self.QValue = util.Counter()#以counter初始化Qvalue,更方便

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.QValue[(state, action)]#回傳當前state進行某個action時的Qvalue

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        '''
        計算所有可以走的action中最大Qvalue的數值，將會用在後面迭代算法中找出下一個state
        的所有action中最大的Qvalue
        '''
        legalActions = self.getLegalActions(state)
        mx_nxt_reward=float('-inf')
        if len(legalActions)==0:
          return 0.0
        for a in legalActions:
          current_position = state
          nxt_reward = self.getQValue(current_position,a)
          if nxt_reward >= mx_nxt_reward:
            mx_nxt_reward = nxt_reward
        return mx_nxt_reward

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        '''
        找出當前所在的位置，走哪一個action可以得到最大的Qvalue
        '''
        mx_nxt_reward=float('-inf')
        best=None
        legalActions = self.getLegalActions(state)
        for a in legalActions:
          current_position = state
          nxt_reward = self.getQValue(current_position,a)
          if nxt_reward >= mx_nxt_reward:
            best = a
            mx_nxt_reward = nxt_reward
        #util.raiseNotDefined()
        return best

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

      '''
      要決定當前位置的下一步應該是哪一個action，但有一定機率action是隨機決定，
      為此設置np.random.uniform(0,1)與傳入的epsilon比較，決定action是隨機遠或者
      依照每個action的Qvalue選出最佳的action
      '''
      # Pick Action
      legalActions = self.getLegalActions(state)
      best = None

      if np.random.uniform(0, 1) <= self.epsilon:
        best = np.random.choice(legalActions)
      else:
        best=self.getPolicy(state)
      return best

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """

        '''
        為了迭代使效果更好，因此更新current state中要進行的action的Qvalue，由傳入的alpha reward discount與
        下一個state中的每個action相比之下最高的Qvalue計算而來
        '''
        QV=self.getQValue(state,action)
        if nextState:
          self.QValue[(state,action)]=(1-self.alpha)*QV+self.alpha*(reward+self.discount*self.getValue(nextState))
        else:

          self.QValue[(state,action)]=(1-self.alpha)*QV+self.alpha*reward
        #util.raiseNotDefined()

    def getPolicy(self, state):
      '''
      當getAction決定是要以擁有最大Qvalue的action為下一步時，就以此function call 
      computeActionFromQValues找出有最大Qvalue的action
      '''
      return self.computeActionFromQValues(state)

    def getValue(self, state):
      """
      回傳當前state每個action中最大的Qvalue
      """
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
