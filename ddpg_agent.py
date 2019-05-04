import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from parameters import *
from model import Actor, Critic
from util import ReplayBuffer, OUNoise


class DDPGAgent():
    
    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        
        # Directory where to save the model
        self.model_dir = os.getcwd() + "/DDPG/saved_models"
        os.makedirs(self.model_dir, exist_ok=True)

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def act(self, states, add_noise=True):
        states = torch.from_numpy(states).float().to(device)

        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            actions += self.noise.sample()
        
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # adds gradient clipping to stabilize learning
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save_model(self):
        torch.save(
            self.actor_local.state_dict(), 
            os.path.join(self.model_dir, 'actor_params.pth')
        )
        torch.save(
            self.actor_optimizer.state_dict(), 
            os.path.join(self.model_dir, 'actor_optim_params.pth')
        )
        torch.save(
            self.critic_local.state_dict(), 
            os.path.join(self.model_dir, 'critic_params.pth')
        )
        torch.save(
            self.critic_optimizer.state_dict(), 
            os.path.join(self.model_dir, 'critic_optim_params.pth')
        )

    def load_model(self):
        """Loads weights from saved model."""
        self.actor_local.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'actor_params.pth'))
        )
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'actor_optim_params.pth'))
        )
        self.critic_local.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'critic_params.pth'))
        )
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'critic_optim_params.pth'))
        )
