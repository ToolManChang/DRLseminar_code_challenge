# DRLseminar_code_challenge
In this project, two agents are tried with Rllib for this task: Soft Actor Critic (SAC) for discrete action space and AlphaZero. The self-implemented notebook version of SAC for discrete action space is still being tested.
## Jupyter notebook implementation of SAC for discerete actions
The Jupyter notebook "SAC_discrete" is implemented on top of: [1] Soft Actor-Critic for Discrete Action Settings - Petros Christodoulou https://arxiv.org/pdf/1910.07207v2.pdf. The implemented algorithm didn't include twin Q network technique. The result is printed in the notebook. The following winning rate is calculated with 100 average episodes.
```bash
[0.95, 0.55, ]
```
## SAC agent (Rllib)
This agent is provided from Rllib 0.8.3, which is tried in the "evaluation.py". Support of SAC agent for discrete action spaces in RLlib is also implemented on top of: [1]
Here we trained the agent with default configurations in 120000 time steps, the results for the four environments are:
```bash
SAC achieved the following win rates:  [1.0, 0.6108786610878661, 1.0, 0.9]
```
The learning curves are shown bellow. 
![learning curve](https://github.com/ToolManChang/DRLseminar_code_challenge/blob/master/DRL_Seminar_BlackJack/learning%20curve.png)
It can be seen that the normal blackjack is the hardest problem as the agent can at most achieve winning rate of around 0.6-0.65. Considering the normal blackjack are quite uncertain, it is not possible to win all the time, so we want to have a estimation on the upper bound of best performance that a agent can get using model-based agent.
## AlphaZero agent (Rllib)
The training for this agent is implemented in "rllib_alpha_agent.py". We only tried Alphazero agent in the noraml Blackjack environment. The environment that supports alphazero agent in rllib has to implement "get_state" and "set_state" methods. The modified environment "BlackjackEnv_zero" is implemented in the "blackjack.py" python file. With this modified environment, the agent is able to access the full state of current environment (rather than only the observation) and builds the monte carlo tree. The config of monte carlo tree search is:
```bash
    "mcts_config": {
        "puct_coefficient": 1.5,
        "num_simulations": 200,
        "temperature": 1.0,
        "dirichlet_epsilon": 0.20,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True,
    },
```
The performance of monte carlo tree search (MCTS) policy is shown in the figure below:
![monte_carlo](https://github.com/ToolManChang/DRLseminar_code_challenge/blob/master/DRL_Seminar_BlackJack/monte_carlo.png)
We can see that MCTS can achieve at most around 0.7 learning rate. As this model-based tree search also has access to the full state of the environment, so it can be seen to "cheat" compared with other agents, so we can guess based on this result that for normal blackjack we can never achieve winning rates higher than 0.75.

Even learning from this MCTS policy, the Alphazero agent can not achieve as good as MCTS does. The learning curve for Alphazero is (evaluation based on output from the policy network):
![alphazero](https://github.com/ToolManChang/DRLseminar_code_challenge/blob/master/DRL_Seminar_BlackJack/alpha_zero.png)
The final policy only achieve winning rate around 0.6, because it has no access to the full state as MCTS does. In this way, SAC still performs better with high uncertainty in the environment.
