from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        #先設定qvalues的資料型態
        self.qvalues={}


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #如果不在規定的動作內回傳0.0 不然回傳回傳qvalues的位置和action
        
        if (state, action) not in self.qvalues:
            return 0.0
        return self.qvalues[(state, action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #在可以動的方向中找到找出最大值，這邊我用它提供的getLegalActions去找可以動的方向
        #接著把個方向的數字存在values 的list裏面
        #最後用max funciton找最大值回傳
        #如果沒有值那就回傳0.0
        values=[]

        for action in self.getLegalActions(state):
            value=self.getQValue(state,action)
            values.append(value)

        if len(values)==0:
            return 0.0
        else:
            return max(values)



    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # 先存一個空的action list
        # 在可以動的方向中找到找出最好的方向，這邊我用它提供的getLegalActions去找可以動的方向
        # 將符合條件的方向存進actions裏面
        # 如果沒有直那就回傳None 反之random其中一個回傳

        actions = []
        for action in self.getLegalActions(state):
            if self.getQValue(state, action) == self.getValue(state):
                actions.append(action)
        if len(actions) != 0:
            return random.choice(actions)
        else:
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
        #計算現在狀態下的每個action，利用程式已給的epsion參數
        #接下來就是看上面寫的去把它給的function代進去
        #主要就是看要不要要隨機選取下一步的方向，不然就用computeactionfromqvalue的那個funciton決定
        #這樣才能學習
        if util.flipCoin(self.epsilon):
            next_action = random.choice(legalActions)
            return next_action
        return self.computeActionFromQValues(state)



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #公式
        #更新數字
        self.qvalues[(state, action)] = self.alpha * (reward + self.discount * self.getValue(nextState)) + (1 - self.alpha) * self.getQValue(state, action)


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)



