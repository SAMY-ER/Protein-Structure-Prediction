# Protein Folding Problem : A Reinforcement Learning Approach

## Description

The following is an AI project that aims to solve a simplified version of the Protein Folding Problem using Reinforcement Learning. The Protein Folding Problem is concerned with finding the *Native State* of a protein given its sequence of amino-acids. When a protein is synthesized, it folds rapidly in the space until it reaches a stable form, usually referred to as the Native State. After this, the protein can occupy its intended function. 

Knowing the outcome of the folding process in advance is one of the major challenges in Bioinformatics, and solving it would have important implications in several domains ranging form biochemistry to genetic engineering.

For a more detailed description of the problem, the task formulation and the proposed solution, check out the project's [paper](https://github.com/SAMY-ER/Protein-Folding-Problem/blob/master/report/Protein%20Folding%20Problem%20-%20A%20Reinforcement%20Learning%20Approach.pdf) !

## Research Questions

Through this project, we will try to answer the following questions :

* How can we cast the problem as a combinatorial optimization task ?
* How do we frame the task from a Reinforcement Learning perspective ? What is the State Space, Action Space and Reward Function ?
* How viable is Reinforcement Learning in solving the Protein Folding Problem ?

##  Installation

In order to install and run this module, complete the following steps. 

1. Create a virtual environment (this step is optional):

**Using virtualenv**
```
>> mkdir ~/envs
>> python -m venv ~/envs/pfpenv
>> source ~/envs/pfpenv/bin/activate
```
**Using conda envs**
```
>> conda create --name pfpenv
>> conda activate pfpenv
```

2. Install requirements:
```
>> cd path/to/Protein-Folding-Problem
>> pip install -r requirements.txt
```

3. Run setup.py:
```
>> pip install .
```

## Usage


<img src="./docs/animated.svg" width="80%" height="60%">

## Components

This section describes the different components of the module : 
    1. The environment
    2. The agents

### 1/ Environment

### 2/ Agents


## Preview

![alt text](./report/figures/predicted_native_state.png "Predicted Native State - sequence : PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP")

![alt text](./report/figures/env_summary.png "Environment Summary")
