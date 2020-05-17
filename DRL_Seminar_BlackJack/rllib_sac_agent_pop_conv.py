from Blackjack import BlackjackEnv
import numpy as np
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from ray import tune
from ray.rllib.agents.sac import SACTrainer


analysis=tune.run(SACTrainer, stop={"timesteps_total": 200000},
                  config={
                      "env": BlackjackEnv,
                      "gamma": 0.99,
                      "num_workers": 1,
                      "num_envs_per_worker": 8,
                      "optimization": {
                          "actor_learning_rate": tune.grid_search([1e-4,3e-4]),
                          "critic_learning_rate": tune.grid_search([1e-4,3e-4]),
                          "entropy_learning_rate": 3e-4,
                      },
                      "evaluation_interval": 1,
                      "evaluation_num_episodes": 50,
                  })
print(analysis.get_best_config("episode_reward_mean"))