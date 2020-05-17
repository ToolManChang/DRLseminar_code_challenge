from gym.envs.registration import register
from .blackjack import BlackjackEnv, BlackjackEnv_zero, BlackjackEnv_conv

register(
    id='Blackjack-v1',
    entry_point='Blackjack:BlackjackEnv'
)

register(
    id='Blackjack-v2',
    entry_point='Blackjack:BlackjackEnv_zero'
)

register(
    id='Blackjack-v3',
    entry_point='Blackjack:BlackjackEnv_conv'
)
