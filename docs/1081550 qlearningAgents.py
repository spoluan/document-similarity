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
        # create a counter for Q-values
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # all keys of counter are defaulted to have value 0.0
        # return Q( s,a )
        return self.QValues[(state,action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # get all actions of this state
        actions = self.getLegalActions(state)

        # no legal actions, terminal state, return 0.0
        if len(actions) == 0 :
            return 0.0

        # compare Q-value of all actions and get the max Q-value
        # max Q-value is defaulted to minimum float
        max_QValue = float('-inf')
        for action in actions :
            # get Q-value of this action
            action_QValue = self.getQValue(state,action)
            if max_QValue < action_QValue :
                max_QValue = action_QValue

        return max_QValue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # best action is max Q-value of this state
        # max Q-value is maybe not only one, create a list for them
        best_action_list = []

        # get all actions of this state
        actions = self.getLegalActions(state)

        # no legal actions, terminal state, return None
        if len(actions) == 0:
            return None

        # compare Q-value of all actions and get the best action(s)
        # max Q-value is defaulted to minimum float
        max_QValue = float('-inf')
        for action in actions:
            # get Q-value of this action
            action_QValue = self.getQValue(state, action)
            if max_QValue < action_QValue:
                # empty best action list
                best_action_list = []

                # put this action in best action list
                best_action_list.append(action)

                max_QValue = action_QValue
            elif max_QValue == action_QValue:
                # put this action in best action list
                best_action_list.append(action)

        # random return the best action from one or more best actions
        return random.choice(best_action_list)


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
        # with small probability, take a random action
        if util.flipCoin(self.epsilon) == True:
            action = random.choice(legalActions)

        # with big probability, take the best policy action
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
        # sample = reward + ( discount ) * Q( s',a' )
        sample = reward + self.discount * self.getValue(nextState)

        # Q(s,a) = ( 1-alpha ) * Q( s,a ) + ( alpha )[ sample ]
        self.QValues[(state,action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

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
        # get all features of this state
        features = self.featExtractor.getFeatures(state,action)

        # Q( s, a ) = w1 f1( s, a ) + w2 f2( s, a ) +...+ wn fn( s, a )
        # __mul__ is the dotProduct operator
        QValue = features.__mul__(self.weight)
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # get all features of this state
        features = self.featExtractor.getFeatures(state, action)

        # difference = [ reward + ( discount ) * Q( s', a' ) ] - Q( s, a )
        difference = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)

        # wi = wi + ( alpha ) * [ difference ] * fi( s, a )
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * difference * features[feature]

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
