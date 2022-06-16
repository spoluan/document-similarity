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


from collections import defaultdict

from flask import current_app
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
        
        """
        # 預設 actions 選項
        actions = ("exit", "north", "east", "south", "west")
        
        # 生成一個 q_table 儲存 key = state，value = [ actions = 0.0 ]。
        self.q_table = defaultdict(lambda: [ 0.0 for action in range(len(actions)) ])

        # 編碼 actions in q_table：{ "exit" : 0 , ... }
        self.q_actions = dict( map(reversed , enumerate(actions)) )
        # print("q_actions:", self.q_actions)
        """
        
        # # 確認 args
        # print("args:", args)
        # actionFn = args["actionFn"]
        # start_state = (0,0)
        # actions = actionFn(start_state)
        # print("actions:" , actions)

        # 預設最多動作數量
        max_actions = 20
        
        # 生成一個 q_table 儲存 key = state / value = [ actions = 0.0 ]。
        self.q_table = defaultdict(lambda: [ 0.0 for action in range(max_actions) ])

        # 編碼 actions in q_table：{ "exit" : 0 , ... }
        self.q_actions = defaultdict()
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # 若是未看過的 action，則進行初始化。
        if action not in self.q_actions:
          self.q_actions[action] = len(self.q_actions)

        # 若是未看過的 state，則進行初始化。
        if state not in self.q_table:
          self.q_table[state][self.q_actions[action]] = 0.0
        
        # 回傳目前 q_table 中的 action_q_value
        return self.q_table[state][self.q_actions[action]]

        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # 取得當下 state 的所有合法動作。
        actions = self.getLegalActions(state)
        # print("LegalActions:", actions)
        
        # 若無合法動作，回傳 0.0。
        if len(actions) == 0:
          return 0.0
        
        # 從合法動作裡面選取 q value 最大值。
        max_value = None
        for action in actions:
          temp_value = self.getQValue(state, action)

          if max_value == None:
            max_value = temp_value
          else:
            if max_value < temp_value:
              max_value = temp_value
        
        return max_value

        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # 取得當下 state 的所有合法動作。
        actions = self.getLegalActions(state)
        # print("LegalActions:", actions)

        # 若無合法動作，回傳 None。
        if len(actions) == 0:
          return None

        # 從合法動作裡面選取 q value 最大值的動作。
        max_action = None
        max_value = None
        for action in actions:
          temp_value = self.getQValue(state, action)

          if max_value == None:
            max_value = temp_value
            max_action = action
          else:
            if max_value < temp_value:
              max_value = temp_value
              max_action = action
        
        return max_action

        # util.raiseNotDefined()

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

        # True == ( r < probability == self.epsilon )，使用隨機走法。
        if (util.flipCoin(self.epsilon)):
          # random action
          action = random.choice(legalActions)
        else:
          # the best policy action
          action = self.getPolicy(state)

        # util.raiseNotDefined()

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

        # old estimate
        old_q = self.getQValue(state, action)
        
        # new sample estimate
        sample = reward + self.discount * self.getValue(nextState)

        # Incorporate the new estimate into a running average
        self.q_table[state][self.q_actions[action]] = (1 - self.alpha) * old_q + self.alpha * sample

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
