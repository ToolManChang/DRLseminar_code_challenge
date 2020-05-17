import gym
from Blackjack import BlackjackEnv, BlackjackEnv_zero
import ray
import numpy as np
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel, ActorCriticModel
import ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer as alpha_zero
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()
from ray.tune import run, sample_from
import random





#model config
class MyDenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.original_space["obs"].shape[0],
                out_features=256), nn.ReLU())
        self.actor_layers = nn.Sequential(
           nn.Linear(in_features=256, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(nn.Linear(in_features=256, out_features=256), nn.ReLU(),
                                           nn.Linear(in_features=256, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None


ray.init()

ModelCatalog.register_custom_model("dense_model", DenseModel)
ModelCatalog.register_custom_model("my_model", MyDenseModel)
env = BlackjackEnv_zero()


#custom evaluation
def on_episode_end(info):
    #evaluation
    #print(info)
    episode = info["episode"]
    policy = info["policy"]["default_policy"]
    ave_rewards = 0.0
    for i in range(200):
        state = env.reset()
        terminal = False
        rewards = 0.0
        while not terminal:
            prior, value = policy.model.compute_priors_and_value(state)
            action= np.random.choice(
                np.arange(2), p=prior)
            state, reward, terminal, info = env.step(action)
            rewards+=reward
            #print(state)
            #print(action)
            # print(info)
            # print(reward)

        ave_rewards+=rewards

    ave_rewards/=100
    print("episode {} ended with length {} and rewards {}".format(
        episode.episode_id, episode.length, ave_rewards))
    episode.custom_metrics["test_rewards"] = ave_rewards



config = {
    "env": BlackjackEnv_zero,
    "rollout_fragment_length": 10,
    "train_batch_size": 500,
    "sgd_minibatch_size": 64,
    "lr": tune.grid_search([5e-5,3e-5,1e-4]),
    "num_workers": 1,
    "num_sgd_iter": 1,
    "mcts_config": {
        "puct_coefficient": 1.5,
        "num_simulations": 200,
        "temperature": 1.0,
        "dirichlet_epsilon": 0.20,
        "dirichlet_noise": 0.03,
        "argmax_tree_policy": False,
        "add_dirichlet_noise": True,
    },
    "ranked_rewards": {
        "enable": False
    },
    "model": {
        "custom_model": "my_model"
    },

    "evaluation_interval": 1,
    "evaluation_num_episodes": 1,
    "evaluation_config": {
        "mcts_config": {
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
        },
        "callbacks":
            {
                "on_episode_end": on_episode_end,
            }

    },
}


#train function
def train_zero(config, reporter):
    agent = AlphaZeroTrainer(config)
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


tune.run(
    train_zero,
    name="pbt_humanoid_test",
    stop={"timesteps_total": 3000000},
    max_failures=0,
    config=config,
)


