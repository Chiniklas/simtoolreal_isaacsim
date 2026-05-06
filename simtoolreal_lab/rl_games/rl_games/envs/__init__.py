

from reference.rl_games.rl_games.envs.connect4_network import ConnectBuilder
from reference.rl_games.rl_games.envs.test_network import TestNetBuilder
from reference.rl_games.rl_games.algos_torch import model_builder

model_builder.register_network('connect4net', ConnectBuilder)
model_builder.register_network('testnet', TestNetBuilder)