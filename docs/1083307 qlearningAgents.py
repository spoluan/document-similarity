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

import random,util,math

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
        #  設定 value(state,action)
        self.qvalues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        
        # 取值
        return self.qvalues[(state, action)]

        #util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        # actiion
        actions = self.getLegalActions(state)
        values = []

        # 此state沒有action
        if len(actions) == 0:
            return 0
        # iterate actions 且取state中所有的qvalue
        else:
            for action in actions:
                values.append(self.getQValue(state, action))
        # get max value
        return max(values)

        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        actions = self.getLegalActions(state)
        allactions = []

        # action == 0 in state
        if len(actions) == 0:
            return None
        else:
            for action in actions:
                allactions.append((self.getQValue(state, action), action))
            #取所有最好的qValue和action pairs
            bestactions = [pair for pair in allactions if pair == max(allactions)]
            # 隨機qValue, action pair(如果best不只一個)
            bestActionPair = random.choice(bestactions)
        return bestActionPair[1]

        #util.raiseNotDefined()

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
        
        #probability epsilon隨機action
        prob = self.epsilon
        if util.flipCoin(prob):
            return random.choice(legalActions)
        #回傳最佳action
        else:
            return self.computeActionFromQValues(state)

        return action

        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        
        # 計算 Q(s,a)
        sa = self.getQValue(state, action)
        # get sample
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        # update and add to qvalues dict-counter
        self.qvalues[(state, action)] = (1 - self.alpha) * sa + self.alpha * sample

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
        

        weights = self.getWeights()
         # get features by identity extractor
        feature = self.featExtractor.getFeatures(state, action)

        # compute dotproduct by features * weights
        dotProduct = feature * weights
        return dotProduct

        #util.raiseNotDefined()


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        
        # differece = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        # if weight vector == 0, started == 0
        if len(weights) == 0:
            weights[(state, action)] = 0
        feature = self.featExtractor.getFeatures(state, action)
        # iterate features and multiply by alpha(learing rate) * difference
        for key in feature.keys():
            feature[key] = feature[key] * self.alpha * difference
        # sum weights to newly scaled features
        weights.__radd__(feature)
        # update weights
        self.weights = weights.copy()

        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print (self.getWeights())
            pass
