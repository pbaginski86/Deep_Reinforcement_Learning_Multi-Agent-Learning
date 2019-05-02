import copy
import os
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import DDPGAgent
from parameters import *
from model import Actor, Critic
from util import OUNoise, ReplayBuffer


class MADDPGAgent:
    """Wrapper class managing different agents in the environment."""

    def __init__(self, num_agents, state_size, action_size):
        """Initialize a MADDPGAgent wrapper.
        Params
        ======
            num_agents (int): the number of agents in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        
        self.agents = [DDPGAgent(state_size, action_size, i+1) for i in range(num_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        
        # Will help to decide when to update the model weights
        self.t_step = 0
        
        # Directory where to save the model
        self.model_dir = os.getcwd() + "/MADDPG/saved_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def reset(self):
        """Resets OU Noise for each agent."""
        for agent in self.agents:
            agent.reset()
            
    def act(self, observations, add_noise=False):
        """Picks an action for each agent given their individual observations 
        and the current policy."""
        actions = []
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for a_i, agent in enumerate(self.agents):
                experiences = self.memory.sample()
                self.learn(experiences, a_i)
            
    def learn(self, experiences, agent_number):
        """Helper to pick actions from each agent for the `experiences` tuple that 
        will be used to update the weights to agent with ID = `agent_number`.
        Each observation in the `experiences` tuple contains observations from each 
        agent, so before using the tuple of update the weights of an agent, we need 
        all agents to contribute in generating `next_actions` and `actions_pred`. 
        This happens because the critic will take as its input the combined 
        observations and actions from all agents."""
        next_actions = []
        actions_pred = []
        states, _, _, next_states, _ = experiences
        
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)
        states = states.reshape(-1, self.num_agents, self.state_size)
        
        for a_i, agent in enumerate(self.agents):
            agent_id_tensor = self._get_agent_number(a_i)
            
            state = states.index_select(1, agent_id_tensor).squeeze(1)
            next_state = next_states.index_select(1, agent_id_tensor).squeeze(1)
            
            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))
            
        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        agent = self.agents[agent_number]
        agent.learn(experiences, next_actions, actions_pred)
            
    def save_model(self):
        """Saves model weights to file."""
        for i in range(self.num_agents):
            torch.save(
                self.agents[i].actor_local.state_dict(), 
                os.path.join(self.model_dir, 'actor_params_{}.pth'.format(i))
            )
            torch.save(
                self.agents[i].actor_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'actor_optim_params_{}.pth'.format(i))
            )
            torch.save(
                self.agents[i].critic_local.state_dict(), 
                os.path.join(self.model_dir, 'critic_params_{}.pth'.format(i))
            )
            torch.save(
                self.agents[i].critic_optimizer.state_dict(), 
                os.path.join(self.model_dir, 'critic_optim_params_{}.pth'.format(i))
            )
    
    def load_model(self):
        """Loads weights from saved model."""
        for i in range(self.num_agents):
            self.agents[i].actor_local.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_params_{}.pth'.format(i)))
            )
            self.agents[i].actor_optimizer.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'actor_optim_params_{}.pth'.format(i)))
            )
            self.agents[i].critic_local.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_params_{}.pth'.format(i)))
            )
            self.agents[i].critic_optimizer.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'critic_optim_params_{}.pth'.format(i)))
            )
    
    def _get_agent_number(self, i):
        """Helper to get an agent's number as a Torch tensor."""
        return torch.tensor([i]).to(device)
