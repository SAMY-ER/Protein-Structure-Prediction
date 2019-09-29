# ============================================================================================================================ #
#                                  PROTEIN FOLDING PROBLEM : REINFORCEMENT LEARNING AGENTS                                     #
# ============================================================================================================================ #




# ================= #
# IMPORT LIBRARIES  #
# ================= #

import numpy as np
from collections import OrderedDict


# =================================== #
# CLASS DEFINITION : Q-LEARNING AGENT #
# =================================== #

class QAgent:
    def __init__(self, state_size, action_size, alpha = 1.0, gamma = 0.9, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.99985):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma                                # discount rate
        self.epsilon = epsilon                            # exploration rate
        self.epsilon_min = epsilon_min                    # minimum exploration rate
        self.epsilon_decay = epsilon_decay                # exploration rate decay
        self.alpha = alpha                                # learning rate
        self.Q = OrderedDict({((0,0),(0,1)) : [0, 0, 0]}) # Initial State : ((0,0), (0,1))

    def act(self, state, policy='egreedy'):
        """Method to get action according to a policy."""
        if policy == 'egreedy' : # e-greedy policy
            # With probability (1-epsilon), take the best action (exploit, Greedy Policy)
            if np.random.uniform(0, 1) > self.epsilon: 
                action = np.argmax(self.Q[tuple(state)])
            # With probability epsilon, take random action (explore)
            else: 
                action = np.random.choice(self.action_size)
        else : # greedy policy
            action = np.argmax(self.Q[tuple(state)])
        return action 
    
    def train(self, state, action, reward, next_state):
        """Method to update the Q entries."""
        # Check if next_state is in Q, if not add it
        if tuple(next_state) not in self.Q.keys():
            self.Q[tuple(next_state)] = [0, 0, 0]
        # Compute Max Q(next_state, :)
        next_Q = self.Q[tuple(next_state)]
        # Update Q entry of Current State
        self.Q[tuple(state)][action] = (1 - self.alpha) * self.Q[tuple(state)][action] + self.alpha * (reward + self.gamma * np.max(next_Q))
        return None