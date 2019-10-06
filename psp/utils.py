# ============================================================================================================================ #
#                                             PROTEIN FOLDING PROBLEM : UTILITIES                                              #
# ============================================================================================================================ #




# ================= #
# IMPORT LIBRARIES  #
# ================= #
from collections import namedtuple, deque
import numpy as np
import random


# ==================================================================================================== #
#                                        HELPER FUNCTIONS                                              #
# ==================================================================================================== #

# MOVING AVERAGE
def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# SOFTMAX FUNCTION
def softmax(x):
    """Computes the softmax value for each element in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# ==================================================================================================== #
#                                        PROTEIN SEQUENCES                                             #
# ==================================================================================================== #

# INITIATE PROTEIN SEQUENCES
    # SEQUENCE 1 : LENGTH = 6  | MINIMUM ENERGY = -2
SEQUENCE_1 = 'HHPPHH'
    # SEQUENCE 2 : LENGTH = 8  | MINIMUM ENERGY = -2
SEQUENCE_2 = 'HPPHPPPH'
    # SEQUENCE 3 : LENGTH = 10 | MINIMUM ENERGY = -3
SEQUENCE_3 = 'HHPHPHPPPH'
    # SEQUENCE 4 : LENGTH = 14 | MINIMUM ENERGY = -6
SEQUENCE_4 = 'HPHPPHHPHPPHPH'
    # SEQUENCE 5 : LENGTH = 20 | MINIMUM ENERGY = -9
SEQUENCE_5 = 'HPHPPHHPHPPHPHHPPHPH'
    # SEQUENCE 6 : LENGTH = 36 | MINIMUM ENERGY = -14
SEQUENCE_6 = 'PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP'


# ==================================================================================================== #
#                                         HELPER CLASSES                                               #
# ==================================================================================================== #

# CLASS DEFINITION : MEMORY
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Memory:
    def __init__(self, capacity=10000):
        """Initializes a Memory Buffer."""
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def __len__(self):
        """Returns number of elements in the buffer."""
        return len(self.buffer)
    
    def push(self, *args):
        """Adds a new transition to the buffer."""
        self.buffer.append(Transition(*args))
        
    def sample(self, batch_size):
        """Sample a batch from the buffer."""
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))        
        return batch