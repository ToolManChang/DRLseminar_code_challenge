# DRLseminar_code_challenge
In this project, two agents are implemented for this task: Soft Actor Critic (SAC) for discrete action space and AlphaZero.
While SAC achieve quite good result for all the 4 evaluation environments, AlphaZero can provide a insight on upperbound of winning rate of the second environment (original Blackjack).
## Setup
As ray 0.8.0 provided in the original setup doesn't support SAC with discrete action space, we use ray 0.8.3 instead. So after install and activate the drl-seminar environment, we will install ray 0.8.3 (or directly modify the version in conda_env.yml):
```bash
conda install pip
pip install ray==0.8.3
```
