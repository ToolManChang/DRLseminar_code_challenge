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

class MyPreprocessorClass(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (2,13)  # can vary depending on inputs

    def transform(self, observation):
        player_obs = observation[0].reshape((4,13))
        player_obs = np.sum(player_obs,axis = 0).reshape((1,13))
        #print(player_obs.shape)
        dealer_obs = np.zeros((1,13))
        dealer_obs[0][observation[1]%13] = 1
        #print(dealer_obs.shape)
        t = random.randint(0,51)
        while observation[0][t]==1 or observation[1]==t:
            t = random.randint(0,51)
        dealer_obs[0][t%13]+=1
        #print(dealer_obs.shape)
        new_obs = np.concatenate([player_obs,dealer_obs],axis = 0)
        return new_obs# return the preprocessed observation

class MyPreprocessorClass2(Preprocessor):
    def _init_shape(self, obs_space, options):
        return (2,13)  # can vary depending on inputs

    def transform(self, observation):
        player_obs = observation[0].reshape((4,13))
        player_obs = np.sum(player_obs,axis = 0).reshape((1,13))
        #print(player_obs.shape)
        dealer_obs = np.zeros((1,13))
        dealer_obs[0][observation[1]%13] = 1
        #print(dealer_obs.shape)

        #print(dealer_obs.shape)
        new_obs = np.concatenate([player_obs,dealer_obs],axis = 0)
        return new_obs# return the preprocessed observation

ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
ModelCatalog.register_custom_preprocessor("my_prep2", MyPreprocessorClass2)

#train function
def train_zero(config, reporter):
    agent = SACTrainer(config)
    #agent.restore("/home/yunke/ray_results/AlphaZero_BlackjackEnv_zero_2020-05-01_22-50-303ae70oaq/checkpoint_1981/checkpoint-1981") #continue training
    #training curriculum, start with phase 0

    episodes = 0
    i = 0
    while True:
        result = agent.train()
        if reporter is None:
            continue
        else:
            reporter(**result)
        if i % 50 == 0: #save every 10th training iteration
            checkpoint_path = agent.save()
            print(checkpoint_path)

        i+=1

analysis=tune.run(train_zero, stop={"timesteps_total": 300000},
                  name="sac_single",

                  config={
                      "env": BlackjackEnv,
                      "use_state_preprocessor": True,
                      "gamma": 0.99,
                      "num_workers": 1,
                      "num_envs_per_worker": 8,
                      "optimization": {
                          "actor_learning_rate": 3e-4,
                          "critic_learning_rate": 3e-4,
                          "entropy_learning_rate": 5e-4,
                      },
                      "model":{
                          "custom_preprocessor": tune.grid_search(["my_prep","my_prep2",{}]),
                      }
                  })
print(analysis.get_best_config("episode_reward_mean"))