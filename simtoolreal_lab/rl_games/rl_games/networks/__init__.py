from reference.rl_games.rl_games.networks.tcnn_mlp import TcnnNetBuilder
from reference.rl_games.rl_games.algos_torch import model_builder

model_builder.register_network('tcnnnet', TcnnNetBuilder)