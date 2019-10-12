# ============================================================================================================================ #
#                             PROTEIN STRUCTURE PREDICTION : REINFORCEMENT LEARNING ENVIRONMENT                                #
# ============================================================================================================================ #




# ================= #
# IMPORT LIBRARIES  #
# ================= #

import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import gym
from collections import OrderedDict
from termcolor import colored
import textwrap


# ============================== #
# CLASS DEFINITION : ENVIRONMENT #
# ============================== #

class ProteinStructureEnv:
    def __init__(self, sequence, valid_state_reward = 0.1, collision_reward=0, fig_size=(6, 6)):
        """Initialize new Environement."""
        # Environment Variables
        self.sequence = sequence.upper()
        self.size = len(sequence)
        self.fig_size = fig_size
        self.valid_state_reward = valid_state_reward
        self.collision_reward = collision_reward
        self.ACTION_TO_STR = {0 : 'Left', 1 : 'Forward', 2 : 'Right'}

        self.state = [(0,0), (0,1)] # Initial state is [(0,0), (0,1)] in order to force an initial direction and reduce state space by factor 1/3
        self.state_nn = np.zeros(self.size - 2) - 1 # Initial state_nn is [-1, -1, ..., -1] (This is another representation of the state based on the actions, used for neural nets)
        self.actions = [] # List of ordered actions taken during an episode
        self.is_collision = False # Checks presence of collision
        self.reward = 0 # Counts reward during an episode
        
        self.action_space = gym.spaces.Discrete(3) # Action space of size 3
        self.loc = (0,1) # Last Position on the chain. Initial value is (0,1) (from initial state [(0,0), (0,1)])
        self.direction = 1 # Initial direction is 'Forward'
        self.iter = 0 # Counts number of iterations or steps in the episode

        
    def reset(self, state_nn=False):
        """Reset Environment to initial values."""
        self.state = [(0,0), (0,1)]
        self.state_nn = np.zeros(self.size - 2) - 1
        self.actions = []
        self.reward = 0
        self.is_collision = False
        self.loc = (0,1)
        self.direction = 1
        self.iter = 0
        if state_nn: return copy.copy(self.state_nn)
        return copy.copy(self.state)

        
    def step(self, action, state_nn=False):
        """Advance the agent to the next state using the input action."""
        # Get Next Location
        self.loc, self.direction = self._get_next_loc(self.loc, self.direction, action)  
        # Check Collision
        if self.loc in self.state:
            self.is_collision = True
        # Update
        self.state.append(self.loc)
        self.state_nn[self.iter] = action
        self.actions.append(action)
        self.iter += 1
        # Set Done Parameter
        done = True if (self.iter+2 == self.size) or self.is_collision else False  
        # Compute Reward of (state, action) pair
        reward = self._compute_reward(done)
        self.reward += reward # adds reward to total reward of episode
        # Fill in Information
        info = {
            'chain_length' : self.iter + 2,
            'seq_length'   : self.size,
            'is_collision' : self.is_collision,
            #'actions'      : [self.ACTION_TO_STR[i] for i in self.actions],
            #'state'        : self.state
        }
        if state_nn: return (copy.copy(self.state_nn), reward, done, info)
        return (copy.copy(self.state), reward, done, info)

    
    def _get_next_loc(self, loc, direction, action):
        """Update Environment with respect to input action."""
        # Update Direction
        if action == 0:
            next_direction = (direction - 1)%4
        elif action == 2:
            next_direction = (direction + 1)%4
        else:
            next_direction = direction        
        # Update Last Location
        if next_direction == 0:
            next_loc = (loc[0]-1, loc[1])
        elif next_direction == 1:
            next_loc = (loc[0], loc[1]+1)
        elif next_direction == 2:
            next_loc = (loc[0]+1, loc[1])
        else:
            next_loc = (loc[0], loc[1]-1)
 
        return next_loc, next_direction

    
    def _is_neighbors(self, tuple_1, tuple_2):
        """Check if two tuples are neighbors on the grid."""
        if abs(tuple_1[0] - tuple_2[0]) + abs(tuple_1[1] - tuple_2[1]) == 1:
            return True      
        return False

    
    def _compute_energy(self):
        """Compute energy at the end of episode."""
        E = 0
        for i in range(self.iter + 2 - 2):
            for j in range(i + 2, self.iter + 2):
                if (self.sequence[i] == 'H') and (self.sequence[j] == 'H') and self._is_neighbors(self.state[i], self.state[j]):
                    E -= 1
        return E

    
    def _compute_reward(self, done):
        """Compute reward of the transition."""
        r = 0
        if self.is_collision: # if new state is not valid (exists already), add collision penalty
            r += self.collision_reward
        elif done: # If we reach the end of the sequence with no collisions, add -energy and valid state reward
            r = r + self.valid_state_reward - self._compute_energy()
        else: # If new state is valid and episode not done yet, add reward for valid step
            r += self.valid_state_reward
        return r
    
    
    def summary(self):
        texts = []
        texts.append(colored('\n{:^90}'.format('ENVIRONMENT SUMMARY') , attrs=['reverse', 'bold']))
        texts.append(colored(' Sequence   :', 'green', attrs=['reverse','bold']) + ' ' + ' '.join([colored(char, 'blue', attrs=['bold']) if char=='P' else colored(char, 'red', attrs=['bold']) for char in self.sequence]))
        texts.append(colored(' Length     :', 'green', attrs=['reverse','bold']) + ' ' + colored(str(self.size), attrs=['bold']))
        texts.append(colored(' Collision  :', 'green', attrs=['reverse','bold']) + ' ' + colored(self.is_collision, attrs=['bold']))
        texts.append(colored(' Energy     :', 'green', attrs=['reverse','bold']) + ' ' + colored(str(self._compute_energy()), attrs=['bold']))
        texts.append(colored(' Reward     :', 'green', attrs=['reverse','bold']) + ' ' + colored(str(round(self.reward, 2)), attrs=['bold']))        
        texts.append(colored(' Iteration  :', 'green', attrs=['reverse','bold']) + ' ' + colored(str(self.iter), attrs=['bold']))                
        actions = colored(' > '.join([colored(self.ACTION_TO_STR[i], 'cyan', attrs=['bold']) if self.ACTION_TO_STR[i] == 'Forward' \
                                                                                                  else colored(self.ACTION_TO_STR[i], 'magenta', attrs=['bold']) if self.ACTION_TO_STR[i] == 'Left' \
                                                                                                  else colored(self.ACTION_TO_STR[i], 'yellow', attrs=['bold']) for i in self.actions]), attrs=['bold'])
        actions = textwrap.wrap(actions, width=200)
        actions = [(14* ' ') + line if ix > 0 else line for ix,line in enumerate(actions)]
        texts.append(colored(' Actions    :', 'green', attrs=['reverse','bold']) + ' ' + '\n'.join(actions))
        texts.append(colored(90*' ', attrs=['reverse', 'bold']))
        return '\n'.join(texts)

    
    def render(self):
        """Visualizes the current state of the environment."""
        H_x = [el[0] for idx, el in enumerate(self.state) if self.sequence[idx]=='H']
        H_y = [el[1] for idx, el in enumerate(self.state) if self.sequence[idx]=='H']
        # Polar amino-acids : coordinates
        P_x = [el[0] for idx, el in enumerate(self.state) if self.sequence[idx]=='P']
        P_y = [el[1] for idx, el in enumerate(self.state) if self.sequence[idx]=='P']
        # Total chain coordinates
        T_x = [el[0] for el in self.state] 
        T_y = [el[1] for el in self.state]
        # Visualize
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.fig_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.patch.set_facecolor('#222831')
        ax.plot(T_x, T_y, '.--', color='white', linewidth=.6)
        ax.scatter(H_x, H_y, s=400, facecolor='tab:red', label='H')
        ax.scatter(P_x, P_y, s=400, facecolor='tab:blue', label='P')
        ax.annotate(s='Start', xy=(0,0), xytext=(0.1, -0.35), color='white', weight='bold')
        ax.annotate(s='End', xy=self.loc, xytext=(self.loc[0]+0.1, self.loc[1]-0.35), color='white', weight='bold')
        plt.xlim(min(T_x)-1, max(T_x)+1)
        plt.ylim(min(T_y)-1, max(T_y)+1)
        ax.legend(markerscale=.5)
        plt.show();
        return None