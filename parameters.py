import torch


RANDOM_SEED = 0
BUFFER_SIZE = int(1e5)  # replay buffer size (1e6 in original paper)
BATCH_SIZE = 1024       # minibatch size (64 in original paper)
GAMMA = 0.9             # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay (1e-2  in original paper)

# Environment Information
NUM_AGENTS = 2
STATE_SIZE = 24
ACTION_SIZE = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")