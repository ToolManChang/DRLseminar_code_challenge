import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box

def sum_hand(card_values):
    sum = np.sum(card_values)
    if 1 in card_values and sum + 10 <= 21:
        return sum + 10  # use ace as 11
    return sum


class BlackjackEnv(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self, one_card_dealer=False, card_values=None):
        self.action_space = spaces.Discrete(2)
        #0,1
        self.observation_space = spaces.Tuple((spaces.MultiBinary(52), spaces.Discrete(52)))
        #(0,1)*52,0-51
        self._card_values = np.asarray(card_values)
        if card_values is None:
            self._card_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4)
        assert len(self._card_values) == 52
        self._one_card_dealer = one_card_dealer
        self.dealer_hand_sum = 1
        self.limit = 21
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.running_reward = 0
        self._deck = np.arange(52)
        #0-51
        #all cards
        self.np_random.shuffle(self._deck)
        #get the first 2 cards
        self._player_cards, self._deck = self._deck[:2], self._deck[2:]
        #another 2 cards
        self._dealer_cards, self._deck = self._deck[:2], self._deck[2:]
        return self._get_obs()

    def _get_obs(self):
        obs = np.asarray([False] * 52)
        obs[self._player_cards] = True
        #can only see the cards player get and the one of the card of dealer.
        return obs, self._dealer_cards[0]

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        #sum the card values
        player_sum = sum_hand(self._card_values[self._player_cards])
        info = {'player hand sum before': player_sum}
        if action:  # hit
            #get one card from the deck
            self._player_cards = np.append(self._player_cards, self._deck[0])
            self._deck = self._deck[1:]
            #update the player_sum
            player_sum = sum_hand(self._card_values[self._player_cards])
        else:
            if self._one_card_dealer:
                #only one card_dealer: if larger than the first of dealer_cards, reward 1
                reward = float(player_sum > self._card_values[self._dealer_cards[0]])
            else:
                #continuously gather cards while sum less than 17
                while sum_hand(self._card_values[self._dealer_cards]) < self.dealer_hand_sum:
                    self._dealer_cards = np.append(self._dealer_cards, self._deck[0])
                    self._deck = self._deck[1:]

                #sum all the values in hand
                dealer_sum = sum_hand(self._card_values[self._dealer_cards])
                info.update({'dealer hand sum': dealer_sum})
                if dealer_sum > 21:
                    reward = 1
                else:
                    reward = float(player_sum > dealer_sum)
        info.update({'action played': 'Hit' if action else 'Stand', 'player hand sum now': player_sum})
        #if players' card value>21 or no action then stop
        self.running_reward+=reward
        done = player_sum > self.limit or not action
        return self._get_obs(), reward, done, info

    def render(self):
        pass

    def set_phase(self,phase):
        if phase==0:
            self._one_card_dealer=1
        if phase==1:
            self._one_card_dealer=0
            self.dealer_hand_sum=8
            self.limit = 30
        if phase==2:
            self.dealer_hand_sum=13
            self.limit = 25
        if phase==3:
            self.dealer_hand_sum=17
            self.limit = 21




class BlackjackEnv_zero:
    def __init__(self, one_card_dealer=False, card_values=None):
        self.env = BlackjackEnv()
        self.action_space = spaces.Discrete(2)
        self.observation_space = Dict({
            "obs": spaces.MultiBinary(104),
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        })
        self.running_reward = 0

    def transform(self,obs):
        #print(obs)
        play_obs = obs[0]
        deal_obs = obs[1]
        new_deal = np.zeros(52)
        new_deal[deal_obs]=1
        return np.array([play_obs,new_deal])

    def reset(self):
        self.running_reward = 0
        obs = self.env.reset()
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}, score, done, info


    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = self.env._get_obs()
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}

    def get_state(self):
        return deepcopy(self.env), self.running_reward


    def set_phase(self,phase):
        self.env.set_phase(phase)
    # def get_state(self):
    #     obs = np.asarray([False] * 52)
    #     obs[self._player_cards] = True
    #     #can only see the cards player get and the one of the card of dealer.
    #     return obs, self._dealer_cards[0]
    #
    # def set_state(self,observation):
    #     #set player cards
    #     self._player_cards = self._card_values[observation[0]]
    #     mask = np.ones(len(self._card_values), np.bool)
    #     mask[observation[0]] = 0
    #     self._deck = np.arange(52)
    #     self._deck = self._deck[mask]
    #     #set dealer card
    #     self._dealer_cards = observation[1]
    #     self._deck = self._deck[0:observation[1],observation[1]:]
    #     #shuffle deck
    #     self.np_random.shuffle(self._deck)
    #     #get another card for dealer
    #     self._dealer_cards = np.append(self._dealer_cards,self._deck[0])
    #     self._deck = self._deck[1:]
    #     #can only see the cards player get and the one of the card of dealer.
    #     return True
class BlackjackEnv_conv:
    def __init__(self, one_card_dealer=False, card_values=None):
        self.env = BlackjackEnv()
        self.action_space = spaces.Discrete(2)
        self.observation_space = Dict({
            "obs": Box(low=0, high=1, shape=(1,8,13)),
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        })
        self.running_reward = 0

    def transform(self,obs):
        #print(obs)
        play_obs = obs[0]
        deal_obs = obs[1]
        new_deal = np.zeros(52)
        new_deal[deal_obs]=1
        return np.array([play_obs,new_deal]).reshape((1,8,13))

    def reset(self):
        self.running_reward = 0
        obs = self.env.reset()
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}, score, done, info


    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = self.env._get_obs()
        return {"obs": self.transform(obs), "action_mask": np.array([1, 1])}

    def get_state(self):
        return deepcopy(self.env), self.running_reward


    def set_phase(self,phase):
        self.env.set_phase(phase)