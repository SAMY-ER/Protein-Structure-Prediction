# ============================================================================================================================ #
#                               PROTEIN STRUCTURE PREDICTION : REINFORCEMENT LEARNING AGENTS                                   #
# ============================================================================================================================ #




# ================= #
# IMPORT LIBRARIES  #
# ================= #

import numpy as np
from collections import OrderedDict
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Huber
from .utils import moving_average, softmax, Memory


# =================================== #
# CLASS DEFINITION : Q-LEARNING AGENT #
# =================================== #

class QAgent:
    def __init__(self, state_size, action_size, alpha = 1.0, gamma = 0.9, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.99985):
        # Parameters
        self.state_size = state_size                      # Size of the state space (not used)
        self.action_size = action_size                    # Size of the action space
        self.gamma = gamma                                # Discount factor
        self.epsilon = epsilon                            # Exploration rate
        self.epsilon_min = epsilon_min                    # Minimum exploration rate
        self.epsilon_decay = epsilon_decay                # Exploration decay rate
        self.alpha = alpha                                # Learning rate of the Q-learning update rule
        self.temperature = 1.0                            # Temperature constant for the Botlzmann Distributed Exploration
        self.temperature_min = 0.01                       # Minimum temperature
        self.temperature_decay = 0.99991                  # Temperature decay rate
        # Q Matrix
        self.Q = OrderedDict({((0,0),(0,1)) : [0, 0, 0]}) # Initial State : ((0,0), (0,1)) 

    def act(self, state, policy='egreedy'):
        """Method to select an action according to a policy."""
        if policy == 'egreedy' :
            # With probability (1-epsilon), take the best action (exploit)
            if np.random.uniform(0, 1) > self.epsilon: action = np.argmax(self.Q[tuple(state)])
            # With probability epsilon, take random action (explore)
            else: action = np.random.choice(self.action_size)
        elif policy == 'boltzmann' :
            # Take action according to boltzmann distribution
            Q_dist = softmax(np.array(self.Q[tuple(state)])/self.temperature)
            action = np.random.choice(range(self.action_size), p=Q_dist) 
        else : # greedy policy
            action = np.argmax(self.Q[tuple(state)])
        return action 
    
    def train(self, state, action, reward, next_state, done):
        """Method to update the Q entries."""
        # Check if next_state is in Q, if not add it
        if tuple(next_state) not in self.Q.keys():
            self.Q[tuple(next_state)] = [0, 0, 0]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[tuple(next_state)])
        # Update Q entry of Current State
        self.Q[tuple(state)][action] = (1 - self.alpha) * self.Q[tuple(state)][action] + self.alpha * target
        return None

    def save(self, fname='../models/model.pkl'):
        """Method to save a Q-Learning agent."""
        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return None
            
    @staticmethod
    def load(fname):
        """Method to load a Q-Learning agent."""
        with open(fname, 'rb') as f:
            print(fname)
            agent = pickle.load(f)           
        return agent


# ============================================= #
# CLASS DEFINITION : DEEP Q-NETWORK (DQN) AGENT #
# ============================================= #

class DQNAgent:
    def __init__(self, input_dim, nb_actions, learning_rate = 0.001, batch_size = 32, memory_capacity=10000, gamma = 0.95, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.99997):
        """Initializes a Deep Q-Network (DQN) Agent."""
        # Parameters
        self.input_dim = input_dim                         # Input dimension of the neural network
        self.nb_actions = nb_actions                       # Number of possibble actions
        self.memory = Memory(capacity=memory_capacity)     # Replay memory
        self.gamma = gamma                                 # Discount factor
        self.epsilon = epsilon                             # Exploration rate
        self.epsilon_min = epsilon_min                     # Minimum exploration rate
        self.epsilon_decay = epsilon_decay                 # Exploration decay rate
        self.learning_rate = learning_rate                 # Learning rate of neural network
        self.batch_size = batch_size                       # Batch size
        # Models
        self.policy_network = self._build_model()
  
    def _build_model(self):
        """Builds architecture of neural network."""
        model = Sequential()
        model.add(Dense(16, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.nb_actions, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(lr=self.learning_rate))
        return model
        
    def remember(self, state, action, next_state, reward, done):
        """Stores memory of experienced events."""
        self.memory.push(state, action, next_state, reward, done)
        
    def act(self, state, policy='egreedy'):
        """Method to select an action according to a policy."""
        if policy == 'egreedy' : # e-greedy policy    
            if np.random.uniform(0, 1) > self.epsilon : # With probability (1-epsilon), take the best action (exploit, Greedy Policy)
                action = np.argmax(self.policy_network.predict(state.reshape(1,-1)).squeeze(0))         
            else : # With probability epsilon, take random action (explore)
                action = np.random.choice(self.nb_actions)
        else : # greedy policy
            action = np.argmax(self.policy_network.predict(state.reshape(1,-1)).squeeze(0))
        return action
    
    def train(self):
        """Trains the policy neural network using transitions randomly sampled from memory."""
        if len(self.memory) < self.batch_size:
            return {'loss':[np.nan]}
        
        batch = self.memory.sample(self.batch_size)
        input_batch = np.vstack(batch.state)
        target_batch = self.policy_network.predict(input_batch)
        next_Q = self.policy_network.predict(np.vstack(batch.next_state))
        ix = np.arange(self.batch_size)
        target_batch[ix, batch.action] = np.array(batch.reward) + (1 - np.array(batch.done)) * self.gamma * np.max(next_Q, axis=1)       
        history = self.policy_network.fit(input_batch, target_batch, batch_size=self.batch_size, epochs=1, verbose=0)
        return history.history

    def save(self, fname='../models/model.h5'):
        """Method to save the Policy Network of a DQN agent."""
        self.policy_network.save(fname)
        return None
    
    @staticmethod
    def load(fname):
        """Method to load a Policy Network in a new DQN agent."""
        agent = DQNAgent(14, 3)
        agent.policy_network = load_model(fname)
        return agent


# ===================================================== #
# CLASS DEFINITION : DOUBLE DEEP Q-NETWORK (DDQN) AGENT #
# ===================================================== #

class DDQNAgent:
    def __init__(self, input_dim, nb_actions, learning_rate = 0.001, batch_size = 64, target_update = 100, memory_capacity=10000, gamma = 0.95, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.99997):
        """Initializes a Double Deep Q-Network (DDQN) Agent."""
        # Parameters
        self.input_dim = input_dim                          # Input dimension of the neural network
        self.nb_actions = nb_actions                      # Number of possibble actions
        self.memory = Memory(capacity=memory_capacity)      #
        self.gamma = gamma             
        self.epsilon = epsilon          
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.target_update = target_update
        # Models
        self.policy_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
       
    def _build_model(self):
        """Builds architecture of neural network."""
        model = Sequential()
        model.add(Dense(16, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.nb_actions, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(lr=self.learning_rate))
        return model
        
    def update_target_network(self):
        """Copy the weights from Model into Target_Model"""
        self.target_network.set_weights(self.policy_network.get_weights())
       
    def remember(self, state, action, next_state, reward, done):
        """Stores memory of experienced events."""
        self.memory.push(state, action, next_state, reward, done)
        
    def act(self, state, policy='egreedy'):
        """Method to select an action according to a policy."""
        if policy == 'egreedy' : # e-greedy policy    
            if np.random.uniform(0, 1) > self.epsilon : # With probability (1-epsilon), take the best action (exploit, Greedy Policy)
                action = np.argmax(self.policy_network.predict(state.reshape(1,-1)).squeeze(0))         
            else : # With probability epsilon, take random action (explore)
                action = np.random.choice(self.nb_actions)
        else : # greedy policy
            action = np.argmax(self.policy_network.predict(state.reshape(1,-1)).squeeze(0))
        return action
   
    def train(self):
        """Trains the neural network using experiences randomly sampled from memory."""
        if len(self.memory) < self.batch_size:
            return {'loss':[np.nan]}
        
        batch = self.memory.sample(self.batch_size)
        input_batch = np.vstack(batch.state)
        target_batch = self.policy_network.predict(input_batch)
        next_Q = self.target_network.predict(np.vstack(batch.next_state))
        ix = np.arange(self.batch_size)
        target_batch[ix, batch.action] = np.array(batch.reward) + (1 - np.array(batch.done)) * self.gamma * np.max(next_Q, axis=1)       
        history = self.policy_network.fit(input_batch, target_batch, batch_size=self.batch_size, epochs=1, verbose=0)
        return history.history

    def save(self, fname='../models/model.h5'):
        """Method to save the Policy Network of a DDQN agent."""
        self.policy_network.save(fname)
        return None
    
    @staticmethod
    def load(fname):
        """Method to load a Policy Network in a new DDQN agent."""
        agent = DDQNAgent(14, 3)
        agent.policy_network = load_model(fname)
        return agent