'''
遇到的問題:
對於變數的了解耗費的許多時間以及搞懂其中概念來如何計算Q值
'''

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
        
        #設初始值
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        #回傳Q(state,action)
        return self.QValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        #計算現在是不是在terminal state
        #IsOverLegal = self.getLegalActions(state)
        
        #如果不是terminal state 就 Return max_action Q(state,action)
        #action是由computeActionFromQValues這個涵式中求出的
        """
        for i in range(self.iterations):
            
            states = self.mdp.getStates()
            temp_counter = util.Counter()
            for state in states:
                if len(self.mdp.getPossibleActions(state))==0:
                    maxVal = 0
                else:
                    maxVal = -float('inf')
                    for action in self.mdp.getPossibleActions(state):
                        Q = self.computeQValueFromValues(state ,action)
                        if Q>maxVal:
                            maxVal = Q
                temp_counter[state] = maxVal
            self.values = temp_counter 
        """
        if len(self.getLegalActions(state)) != 0:
            return self.getQValue(state, self.getPolicy(state))
        else:#剩下如果沒有合法行為則視為terminal state 然後回傳0.0        
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        if len(self.getLegalActions(state)) != 0:#nonternmial state
            Actions = self.getLegalActions(state) #儲存所有合法行為
            aVals = {} #儲存該行為的QValue
            best = float('-inf') 
            
            #跑所有行為計算每個行為的QValue
            for action in Actions:
                temp = self.getQValue(state, action)#計算該行為的QValue
                aVals[action] = temp #放到aVals中
                
            #找最大的QValue 
            for i in aVals:
                if aVals[i] > best:
                    best = aVals[i]
            
            #求出best acion
            bestActions = [k for k,v in aVals.items() if v == best]
            
            return random.choice(bestActions)
            
        else:#ternmial state是就回傳None
            return None
        
        
        

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
        
        #terminal state 回傳None
        if len(legalActions) == 0:
            return None
        
        #照上面要求如果 probability self.epsilon, 就採 random action
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:#不然就用best policy action
            action = self.getPolicy(state)
            
        #util.raiseNotDefined()
        
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
        
        before = self.getQValue(state,action)#原本QVale
        future = self.getValue(nextState)#未來QValue
        
        #計算新的QValue
        temp = reward + (self.discount * future) 
        new = self.alpha * (temp - before)
        
        #更新新的QValue
        self.QValues[(state, action)] = before + new

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

    def getWeights(self):#回傳weight
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        #featureVector儲存 (由食物和牆壁位置和鬼的位置)所計算出的數值
        #Feature extractors for Pacman game states
        featureVector = self.featExtractor.getFeatures(state, action)
        #乘上權重將ans回傳
        ans = featureVector * self.weights
        return ans

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        #features儲存(由食物和牆壁位置和鬼的位置)所計算出的數值
        features = self.featExtractor.getFeatures(state, action)
        #將原始QVale更新為新的QVale
        before = self.getQValue(state, action)#初始QVale
        future = self.getValue(nextState)#更新後QVale        
        
        Weights=[]#儲存每一個feature的weight
        
        #計算每個feature的weight
        for feature in features:
            Weights[feature] = (reward + self.discount * future) - before
            
        #Update每個weight
        for i in features:
            self.weights[i] += self.alpha * Weights[i] * features[i]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass