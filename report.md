# DRLND Project 3 Report (Collaboration and Competition)

## Introduction

After using project 2 in combination with a Soft-Actor-Critic method that wasn't performing well, then converting back to a DDPG algorithm, I decided to now stay in the DDPG family. I have this time used the same solution, DDPG. The basic implementation is very similar to the implementation in the paper referenced below. Additionally, this implementation resembles that of many other DDPG solutions that are mostly identical in how the tensors flow through to a selected action.

All information on parameters and model outputs can be found in the respective files in the repository. Please see below a plot of scores per episode, demonstarting the 0.5 average over 100 consecutive episodes.

![DDPG Plot of Rewards](https://github.com/pbaginski86/Deep_Reinforcement_Learning_Multi-Agent-Learning/blob/master/download.png)

Lastly, for future work:
Additional work should go towards implementing the newest version of the Soft-Actor-Critic algorithm
to properly compare its performance with DDPG. Some particular parts to be included are:
1. Prioritized Experience Replay: One of the possible improvements already acknowledged in
the original research lays in the way experience is used. When treating all samples the
same, one is not using the fact that one can learn more from some transitions than from
others. Prioritized Experience Replay (PER) is one strategy that tries to leverage this fact by
changing the sampling distribution.
2. Noise Addition: While the DDPG implementation follows this approach, the current late
2018 implementation of the SAC algorithm does not include the OUNoise process.
3. Batch Normalization: In order to make training of the neural networks more efficient, batch
normalization should also be added to the SAC neural networks.
