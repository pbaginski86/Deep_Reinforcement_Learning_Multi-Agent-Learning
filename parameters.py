import torch


RANDOM_SEED = 0                 # random seed for PyTorch, NumPy and random
BUFFER_SIZE = int(1e6)          # replay buffer size
BATCH_SIZE = 512                # minibatch size
GAMMA = 0.99                    # discount factor
TAU = 5e-2                      # for soft update of target parameters
LR_ACTOR = 5e-4                 # learning rate of the actor 
LR_CRITIC = 5e-4                # learning rate of the critic
WEIGHT_DECAY = 0.0              # L2 weight decay
UPDATE_EVERY = 2                # weight update frequency
NOISE_AMPLIFICATION = 1         # exploration noise amplification
NOISE_AMPLIFICATION_DECAY = 1   # noise amplification decay

# Environment Information
NUM_AGENTS = 2
STATE_SIZE = 24
ACTION_SIZE = 2

# PyTorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
