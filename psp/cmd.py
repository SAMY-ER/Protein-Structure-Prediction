"""
PSP - Protein-Structure-Prediction

Usage:
    psp-cli train <agent> <model-file> [--seq=<sequence> --episodes=<episodes>]
    psp-cli run <agent> <model-file> [--seq=<sequence>]
    psp-cli (-h | --help)

Arguments:
    <agent>          Type of agent to train.
    <model-file>     Saved model file.

Options:
    --seq=<sequence>           Protein sequence to train on. [default: HHPPHH]
    --episodes=<episodes>      Number of episodes to train for. [default: 50000]
    -h --help                  Show this screen.
"""

from docopt import docopt
import numpy as np
from psp import ProteinStructureEnv, QAgent, DQNAgent, DDQNAgent, SEQUENCE_1, SEQUENCE_2, SEQUENCE_3, SEQUENCE_4, SEQUENCE_5, SEQUENCE_6
from tensorflow.keras import backend as K

# ================================= #
# FUNCTION DEFINITION : TRAIN AGENT #
# ================================= #

def train_agent(agent, model_file, sequence=SEQUENCE_2, episodes=50000):
    # CREATE ENVIRONMENT
    env = ProteinStructureEnv(sequence, valid_state_reward = 0.1, collision_reward = 0) 

   # GET STATE AND ACTION SPACE SIZE
    state_size = env.size - 2
    nb_actions = env.action_space.n

    if agent == 'dqn':
        # CREATE AGENT
        agent = DQNAgent(state_size, nb_actions, learning_rate=0.001, epsilon_decay=np.exp(np.log(0.05)/episodes))
        # TRAIN AGENT
        rewards = []
        losses = []
        print('\nTRAINING STARTED ...')
        for episode in range(1, episodes+1):
            state = env.reset(state_nn=True)
            done = False
            while not done:
                # Pick Next Action (via e-greedy policy)
                action = agent.act(state, policy='egreedy') 
                # Advance to next step
                next_state, reward, done, info = env.step(action, state_nn=True) 
                 # Remember Experience
                agent.remember(state, action, next_state, reward, done)
                state = next_state
            # Update agent
            history = agent.train()
            if episode % 500 == 0 : 
                print('   Episode: {}/{} - Loss: {:.3f} - Reward: {:.2f} - Epsilon: {:.2f}.'.format(episode, episodes, history['loss'][0], env.reward, agent.epsilon))        
                K.clear_session()
            if agent.epsilon >= agent.epsilon_min : agent.epsilon *= agent.epsilon_decay
            rewards.append(env.reward)
            losses.append(history['loss'][0])
        print('TRAINING FINISHED!')

    elif agent == 'ddqn':
        # CREATE AGENT
        agent = DDQNAgent(state_size, nb_actions, learning_rate=0.001, epsilon_decay=np.exp(np.log(0.05)/episodes))
        # TRAIN AGENT
        rewards = []
        losses = []
        print('\nTRAINING STARTED ...')
        for episode in range(1, episodes+1):
            state = env.reset(state_nn=True)
            done = False
            while not done:
                # Pick Next Action (via e-greedy policy)
                action = agent.act(state, policy='egreedy') 
                # Advance to next step
                next_state, reward, done, info = env.step(action, state_nn=True) 
                 # Remember Experience
                agent.remember(state, action, next_state, reward, done)
                state = next_state
            # Update agent
            history = agent.train()
            if episode % 500 == 0 : 
                print('   Episode: {}/{} - Loss: {:.3f} - Reward: {:.2f} - Epsilon: {:.2f}.'.format(episode, episodes, history['loss'][0], env.reward, agent.epsilon))        
                K.clear_session()
            if agent.epsilon >= agent.epsilon_min : agent.epsilon *= agent.epsilon_decay
            if episode % agent.target_update == 0: agent.update_target_network()
            rewards.append(env.reward)
            losses.append(history['loss'][0])
        print('TRAINING FINISHED!')

    else:
        # CREATE AGENT
        agent = QAgent(state_size, nb_actions, alpha=0.8, epsilon_decay=np.exp(np.log(0.05)/episodes))
        # TRAIN AGENT
        rewards = []
        print('\nTRAINING STARTED ...')
        for episode in range(1, episodes+1):
            if episode % 10000 == 0 : print("  Episode :", episode)
            state = env.reset()
            done = False
            while not done:
                # Pick Next Action (via e-greedy policy)
                action = agent.act(state, policy='egreedy') 
                # Advance to next step
                next_state, reward, done, info = env.step(action) 
                # Update agent
                agent.train(state, action, reward, next_state, done)
                state = next_state
            
            if agent.epsilon >= agent.epsilon_min : agent.epsilon *= agent.epsilon_decay
            rewards.append(env.reward)
        print('TRAINING FINISHED!')

    # SAVE AGENT
    if model_file:
        print('SAVING MODEL ...', end=' ')
        try:
            agent.save(fname=model_file)
        except:
            if agent == 'q': 
                agent.save(fname='model.pkl') 
                print('Problem occured, saving the model under /model.pkl ...', end=' ')
            else: 
                agent.save(fname='model.h5')
                print('Problem occured, saving the model under /model.h5 ...', end=' ')
        print('DONE!')
    return None


# =============================== #
# FUNCTION DEFINITION : RUN AGENT #
# =============================== #

def run_agent(agent, model_file, sequence=SEQUENCE_2):
    # CREATE ENVIRONMENT
    env = ProteinStructureEnv(sequence, valid_state_reward = 0.1, collision_reward = 0)

    # LOAD AGENT
    if agent == 'q':
        agent = QAgent.load(model_file)
        state_nn = False
    elif agent == 'dqn':
        agent = DQNAgent.load(model_file)
        state_nn = True
    elif agent == 'ddqn':
        agent = DDQNAgent.load(model_file)
        state_nn = True
    else:
        raise ValueError('The value entered for the argument agent is wrong. Please choose among ["q", "dqn", "ddqn"]')        

    state = env.reset(state_nn=state_nn)
    done = False
    while not done:
        # Pick Next Action (via greedy policy)
        action = agent.act(state, policy='greedy')
        # Advance to next step
        state, reward, done, info = env.step(action, state_nn=state_nn)
    print(env.summary())
    env.render()
    return None


# ========================== #
# FUNCTION DEFINITION : MAIN #
# ========================== #

def main():
    arguments = docopt(__doc__)

    if arguments['train']:
        train_agent(
            arguments['<agent>'],
            arguments['<model-file>'],
            arguments['--seq'],
            int(arguments['--episodes'])
        )
    elif arguments['run']:
        run_agent(
            arguments['<agent>'],
            arguments['<model-file>'],
            arguments['--seq']
        )
    return None



if __name__ == '__main__':
    main()