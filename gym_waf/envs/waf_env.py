import re
import numpy as np
import pandas as pd
import gym
import random
import const
import logging
from gym_waf.envs.features import SqlFeatureExtractor

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}

SEED = 0


class WafEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, payloads_file, csic_file, maxturns=20, turn_penalty=0.1):
        """
        Base class for WAF env
        :param payloads: a list of payload strings
        :param maxturns: max mutation before env ends
        """
        self.action_space = gym.spaces.Discrete(len(ACTION_LOOKUP))
        self.maxturns = maxturns
        self.feature_extractor = SqlFeatureExtractor()
        print("Feature vector shape: {}".format(self.feature_extractor.shape))
        self.history = []
        self.payload_list = None
        # self.max_reward = const.WAF_POSITIVE + 1.0 # waf evasion positive reward + dicriminator real value
        # self.min_reward = const.WAF_NEGATIVE - 1.0 # waf evasion negative reward + dicriminator fake value
        self.max_reward = const.WAF_POSITIVE + self.maxturns*0.1
        self.min_reward = -self.maxturns*0.1
        self.orig_payload = None
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.feature_extractor.shape, dtype=np.float32)
        self.turn_penalty = turn_penalty

        self.payload = None
        self.pre_payload = None
        self.observation = None
        self.turns = 0
        
        self.check_discriminator = const.CHECK_DISCRIMINATOR

        self._load_payloads(payloads_file)
        self._load_csic_http_traffic(csic_file)

    def _load_payloads(self, filepath):
        try:
            with open(filepath, 'r', encoding='UTF-8') as f:
                self.payload_list = f.read().splitlines()
                print("{} payloads dataset loaded".format(len(self.payload_list)))
        except OSError as e:
            logging.error("failed to load dataset from {}".format(filepath))
            raise

    def _load_csic_http_traffic(self, filepath):
        try:
            def normalize_url(url):
                url = re.sub('jsp', 'html', url)
                url = re.sub('tienda1\/', '', url)

                return url

            df = pd.read_csv(filepath)
            df = df[(df.Type == 'Normal')]
            self.csic_http_traffic_list = []
            for index, item in df.iterrows():
                tmp_item_list = []
                for col in list(df.columns):
                    if col == 'URL':
                        item[col] = normalize_url(item[col])

                    tmp_item_list.append(col + ':' + str(item[col]))
                self.csic_http_traffic_list.append(','.join(tmp_item_list))
                
            self.query_parameter_list = ['id', 'nombre', 'precio', 'cantidad', 'B1']
                
            print("{} csic http traffic dataset loaded".format(len(self.csic_http_traffic_list)))
        except OSError as e:
            logging.error("failed to load dataset from {}".format(fileopath))

    def step(self, action_index):
        raise NotImplementedError("_step not implemented")

    def _check_sqli(self, payload):
        raise NotImplementedError("_check_sqli not implemented")

    def _take_action(self, action_index):
        assert action_index < len(ACTION_LOOKUP)
        action = ACTION_LOOKUP[action_index]
        logging.debug(action.__name__, action_index)
        self.history.append(action)
        self.pre_payload = self.payload
        self.payload = action(self.payload, seed=SEED)

    def _process_reward(self, reward):
        reward = reward - self.turns * self.turn_penalty  # encourage fewer turns
        reward = max(min(reward, self.max_reward), self.min_reward)
        return reward

    def reset(self):
        self.turns = 0

        while True:     # until find one that is SQLi by the interface
            payload = random.choice(self.payload_list)
            _ = self._check_sqli(payload)
            if not self.label:
                self.orig_payload = self.payload = payload
                break

        logging.debug("reset payload: {}".format(self.payload))

        self.observation = self.feature_extractor.extract(self.payload)

        return self.observation

    def render(self, mode='human', close=False):
        pass
