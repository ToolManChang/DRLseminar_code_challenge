# DRLseminar_code_challenge
In this project, two agents are implemented for this task: Soft Actor Critic (SAC) for discrete action space and AlphaZero.
While SAC achieve quite good result for all the 4 evaluation environments, AlphaZero can provide a insight on upperbound of winning rate of the second environment (original Blackjack).
## Setup
As ray 0.8.0 provided in the original setup doesn't support SAC with discrete action space, we use ray 0.8.3 instead. So after install and activate the drl-seminar environment, we will install ray 0.8.3 (or directly modify the version in conda_env.yml):
```bash
conda install pip
pip install ray==0.8.3
pip install ray[rllib]==0.8.3
```
## SAC agent
Support of SAC agent for discrete action spaces in RLlib is implemented on top of: [1] Soft Actor-Critic for Discrete Action Settings - Petros Christodoulou https://arxiv.org/pdf/1910.07207v2.pdf
Here we trained the agent with default configurations in 120000 time steps, the results for the four environments are:
```bash
SAC achieved the following win rates:  [1.0, 0.6108786610878661, 1.0, 0.9]
```
The learning curve is shown bellow, where the gray, 
![Learning curve](https://github.com/ToolManChang/DRLseminar_code_challenge/blob/master/DRL_Seminar_BlackJack/Screenshot%20from%202020-05-17%2012-31-59.png)
