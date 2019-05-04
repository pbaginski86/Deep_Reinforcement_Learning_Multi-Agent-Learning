import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import copy
from collections import namedtuple, deque
from parameters import *


def train(ddpg, n_episodes=1000, max_t=1000, save_every=50):
    '''widget = [
        "Episode: ", pb.Counter(), '/' , str(n_episodes), ' ', 
        pb.Percentage(), ' ', pb.ETA(), ' ', pb.Bar(marker=pb.RotatingMarker()), ' ', 
        'Rolling Average: ', pb.FormatLabel('')
    ]
    timer = pb.ProgressBar(widgets=widget, maxval=n_episodes).start()'''

    solved = False
    scores_total = []
    scores_deque = deque(maxlen=100)
    rolling_score_averages = []
    best_score = 0.0
    
    for i_episode in range(1, n_episodes+1):
        current_average = 0.0 if i_episode == 1 else rolling_score_averages[-1]
            
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations[:, -STATE_SIZE:]
        scores = np.zeros(num_agents)
        ddpg.reset()
        
        # for t in range(max_t):
        while True:
            actions = maddpg.act(states)
            
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations[:, -STATE_SIZE:]
            rewards = env_info.rewards
            dones = env_info.local_done
            
            ddpg.step(states, actions, rewards, next_states, dones)
            
            scores += rewards
            states = next_states
            
            if np.any(dones):
                break
        
        max_episode_score = np.max(scores)
        
        scores_deque.append(max_episode_score)
        scores_total.append(max_episode_score)

        average_score = np.mean(scores_deque)
        rolling_score_averages.append(average_score)
        
        if average_score > best_score:
            best_score = average_score
            if solved:
                # This model is better than a previously saved model, so we'll save it
                ddpg.save_model()
        
        if average_score >= 0.5 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, average_score
            ))
            solved = True
            ddpg.save_model()
        
        if i_episode % save_every == 0 and not solved:
            ddpg.save_model()

    ddpg.save_model()

    return scores_total, rolling_score_averages


def plot_results(scores, rolling_score_averages):
    """Plots training results from the training loop in the `train` method.
    Params
    ======
        scores (list): list of the max among all agents in a given episode
        rolling_score_averages (list): average of max agent scores in the last 100 episodes
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(1, len(scores) + 1), scores, label="Max Score")
    plt.plot(np.arange(1, len(rolling_score_averages) + 1), rolling_score_averages, label="Rolling Average")

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(RANDOM_SEED)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(RANDOM_SEED)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(loc=0, scale=1) for _ in range(len(x))])
        self.state = x + dx
        return self.state

 