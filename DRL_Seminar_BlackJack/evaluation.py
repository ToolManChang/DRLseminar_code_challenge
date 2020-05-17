
import numpy as np
environment_configurations = [{"one_card_dealer": True},
                              {},
                              {"card_values": np.ones(52,)*2},
                              {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
                                               4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
                                               2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
                                               5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]

from Blackjack import BlackjackEnv
import numpy as np
import ray
from ray.rllib.utils import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
import random

tf = try_import_tf()
from ray import tune
from ray.rllib.agents.sac import SACTrainer



achieved_win_rates = []
for env_config in environment_configurations:
    print(env_config)
    tune_results = tune.run(SACTrainer,
                            stop={"timesteps_total": 120000},
                            name="sac_single",
                            config={"env": BlackjackEnv,
                                    "env_config": env_config,
                                    "gamma": 0.99,
                                    "num_workers": 1,
                                    "num_envs_per_worker": 8,
                                    "optimization": {
                                        "actor_learning_rate": 3e-4,
                                        "critic_learning_rate": 3e-4,
                                        "entropy_learning_rate": 3e-4,
                                    },
                                    })
    achieved_win_rates.append(tune_results.trials[0].last_result["episode_reward_mean"])

print('SAC achieved the following win rates: ', achieved_win_rates)
# OUTPUT:
#SAC achieved the following win rates:  [1.0, 0.6108786610878661, 1.0, 0.9]
