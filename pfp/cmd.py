"""
PFP - Protein-Folding-Problem

Usage:
    pfp-cli train <agent> <model-file> [--seq=<sequence> --episodes=<episodes>]
    pfp-cli run <model-file> [--seq=<sequence>]
    pfp-cli (-h | --help)

Arguments:
    <agent>          Type of agent to train.
    <model-file>     Saved model file.

Options:
    --seq=<sequence>           Protein sequence to train on. [default: seq.SEQUENCE_2]
    --episodes=<episodes>      Number of episodes to train for. [default: 50000]
    -h --help                  Show this screen.
"""

from docopt import docopt
from pfp import ProteinFoldingEnv, QAgent, SEQUENCE_1, SEQUENCE_2, SEQUENCE_3, SEQUENCE_4, SEQUENCE_5, SEQUENCE_6


# ================================= #
# FUNCTION DEFINITION : TRAIN AGENT #
# ================================= #

def train_agent(agent, model_file, sequence=SEQUENCE_2, episodes=50000):
    # CREATE ENVIRONMENT
    env = ProteinFoldingEnv(sequence, valid_state_reward = 0.1, collision_reward = -1) 

    # CREATE AGENT
    state_size = env.size - 2
    action_size = env.action_space.n
    agent = QAgent(state_size, action_size, epsilon_decay=0.999925, alpha=0.8)

    # TRAIN AGENT
    rewards = []
    print('TRAINING STARTED ...')
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
        agent.save(fname=model_file)
        print('DONE!')
    return None


# =============================== #
# FUNCTION DEFINITION : RUN AGENT #
# =============================== #

def run_agent(model_file, sequence=SEQUENCE_2):
    # CREATE ENVIRONMENT
    env = ProteinFoldingEnv(sequence, valid_state_reward = 0.1, collision_reward = -1)

    # LOAD AGENT
    agent = QAgent.load(model_file)

    state = env.reset()
    done = False
    while not done:
        # Pick Next Action (via greedy policy)
        action = agent.act(state, policy='greedy')
        # Advance to next step
        state, reward, done, info = env.step(action)
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
            arguments['<model-file>'],
            arguments['--seq']
        )
    return None



if __name__ == '__main__':
    main()