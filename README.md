# reinforcement-learning

## Contents
`Shadow-Gym` - template used to create a Gym environment for the Shadow Hand.

`urdfs` - files describing the shape, inertia and mass of various components of the hand

`Shadow-Gym/shadow_gym/envs/shadow_env.py` -  main gymnasium model that interacts with Pybullet. You will mainly modify this file alongside `cube.py` and `hand.py`.

`logs` - each folder in `logs` contains the tensorboard logs for each run.

`models` - each folder containes a trained model named according to number of timesteps used e.g. 30000.zip was trained for 30000 timesteps.

`train.py` - script used to train a model. Automatically generates logs to `logs` and saves model to `models`

`visualise_model.py` - script used to open Pybullet window and view the AI model in action

`evaluate_model.py` - runs a model 100 times and counts number of times it successfully rotates the cube to the target orientation

`Shadow-Gym/shadow-gym/resources/cube_colour_visualiser.py` - utility script to view the hand environment with gravity turned off

### Installation Instructions
(Recommended) Create a virtual environment with conda or venv  

Enter the `Shadow-Gym` folder (folder with setup.py)  
Run `pip install -e .` to install other required packages for the hand model

For the AI modules run
`pip install stable_baselines3 tensorboard`

