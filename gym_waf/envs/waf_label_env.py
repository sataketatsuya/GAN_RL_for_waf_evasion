from termcolor import colored
import logging
import const

from gym_waf.envs.interfaces import ClassificationFailure
from .waf_env import WafEnv

from gym_waf.envs.controls import sqlfuzzer as manipulate
ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.strategies)}


class WafLabelEnv(WafEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_label_waf(self, payload):
        raise NotImplementedError("_get_label not implemented")

    def _get_score_discriminator(self, fake_payload, done=False):
        raise NotImplementedError("_get_label not implemented")

    def _check_sqli(self, payload):
        self.label = self._get_label_waf(payload)

    def step(self, action_index):
        assert self.orig_payload is not None, "please reset() before step()"

        self.turns += 1
        self._take_action(action_index)

        self.observation = self.feature_extractor.extract(self.payload)

        win = False
        reward = 0.0
        self._check_sqli(self.payload)

        if self.label:
            reward = const.WAF_POSITIVE
            episode_over = True
            win = True
            print("WIN with payload: {}".format(colored(repr(self.payload), 'green')))
        elif self.turns >= self.maxturns:
            # out of turns :(
            reward = const.WAF_NEGATIVE
            episode_over = True
        else:
            # reward += 0.0 if self.pre_payload == self.payload else 0.5
            episode_over = False

        if self.check_discriminator:
            discriminator_score = self._get_score_discriminator(self.payload, done=episode_over)
            reward = reward * discriminator_score

        reward = self._process_reward(reward)

        if episode_over:
            logging.debug("episode is over: reward = {}!".format(reward))

        return self.observation, reward, episode_over, \
            {"win": win, "original": self.orig_payload, "payload": self.payload, "history": self.history}

