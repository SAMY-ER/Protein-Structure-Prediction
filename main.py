# ============================================================================================================================ #
#                                            PROTEIN FOLDING PROBLEM : MAIN PROGRAM                                            #
# ============================================================================================================================ #




# ================= #
# IMPORT LIBRARIES  #
# ================= #

from agents import QAgent
from rl_env import ProteinFoldingEnv


# ============ #
# MAIN PROGRAM #
# ============ #

if __name__ == '__main__':
    #with open('/Users/Samyer/Documents/Projects/Protein-Folding-Problem/agent.pickle', 'rb') as handle:
    #    qAgent = pickle.load(handle)
    #print('IMPORTED AGENT !')

    # INITIATE SEQUENCES
    seq1 = 'HHPPHH' # target energy : -2 | target reward : 2.4
    seq2 = 'HPPHPPPH' # target energy : -2 | target reward : 2.6
    seq3 = 'HHPHPHPPPH' # target energy : -3 | target reward : 3.8
    seq4 = 'HPHPPHHPHPPHPH' # target energy : -6 | target reward : 7.2
    seq5 = 'HPHPPHHPHPPHPHHPPHPH' # target energy : -9 | target reward : 10.8
    seq6 = 'PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP' # target energy : -14 | target reward : 17.4

    # CREATE ENVIRONMENT
    env = ProteinFoldingEnv(seq1, valid_state_reward = 0.1, collision_reward = -1) 

    # CREATE Q-LEARNING AGENT
    state_size = env.size - 2
    action_size = env.action_space.n
    qAgent = QAgent(state_size, action_size, epsilon_decay=0.999925, alpha=0.8)
    #9999985 == 2,000,000 #.9999965 == 1,000,000 #.999985 == 200,000 #.999925 == 50,000

    # TRAIN AGENT
    EPISODES = 50000
    qRewards = []
    print('\nTraining Started ...')
    for episode in range(1, EPISODES+1):
        if episode%10000 == 0 : print("  Episode :", episode)
        state = env.reset()
        done = False
        while not done:
            # Pick Next Action (via e-greedy policy)
            action = qAgent.act(state, policy='egreedy') 
            # Advance to next step
            next_state, reward, done, info = env.step(action) 
            # Update Q entries
            qAgent.train(state, action, reward, next_state)
            state = next_state
        
        if qAgent.epsilon >= qAgent.epsilon_min : qAgent.epsilon *= qAgent.epsilon_decay
        qRewards.append(env.reward) 
    print('\nTraining Finished !')

    # TEST AGENT
    state = env.reset()
    done = False
    while not done:
        # Pick Next Action (via greedy policy)
        action = qAgent.act(state, policy='greedy')
        # Advance to next step
        state, reward, done, info = env.step(action)
    print(env.summary())
    env.render()
   
    