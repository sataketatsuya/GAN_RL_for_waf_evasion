from gym.envs.registration import register
import os
import const

MAXTURNS = const.MAXTURNS
TURN_PENALTY = const.TURN_PENALTY
DATASET = os.path.join(os.path.dirname(__file__), 'data', 'sqli-1k.csv')
DATASET_single = os.path.join(os.path.dirname(__file__), 'data', 'sqli-1.csv')
CSIC_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'csic_database.csv')

register(
    id='WafLibinj-single-v0',
    entry_point='gym_waf.envs:LibinjectionEnv',
    kwargs={
        'payloads_file': DATASET_single,
        'csic_file': CSIC_DATASET,
        'maxturns': MAXTURNS,
        'turn_penalty': TURN_PENALTY,
    }
)

register(
    id='WafLibinj-v0',
    entry_point='gym_waf.envs:LibinjectionEnv',
    kwargs={
        'payloads_file': DATASET,
        'csic_file': CSIC_DATASET,
        'maxturns': MAXTURNS,
        'turn_penalty': TURN_PENALTY,
    }
)