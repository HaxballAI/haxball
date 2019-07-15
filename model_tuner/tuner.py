from gym_haxball import onevfixedgym

from random import randrange

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog

import ray.rllib.agents.ppo as ppo

class CustomModel(Model):
    # Example of a custom model.
    # This model just delegates to the built-in fcnet.

    def _build_layers_v2(self, input_dict, num_outputs, options):
        self.obs_in = input_dict["obs"]
        self.fcnet = FullyConnectedNetwork(input_dict, self.obs_space,
                                           self.action_space, num_outputs,
                                           options)
        return self.fcnet.outputs, self.fcnet.last_layer


def tuner():
    ray.init()
    ModelCatalog.register_custom_model("my_model", CustomModel)
    return tune.run(
        "PPO",
        checkpoint_at_end = True,
        stop = {
            "timesteps_total": 10000,
        },
        config = {
            "env" : onevfixedgym.DuelFixedGym,  # or "corridor" if registered above
            "model" : {
                "custom_model" : "my_model",
            },
            "lr" : grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers" : 1,  # parallelism
            "env_config" : {
                "corridor_length" : 5,
                "opponent" : tune.function(lambda x : (randrange(9), randrange(2))),
            },
        },
    )

def getagent():
    ModelCatalog.register_custom_model("my_model", CustomModel)
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["env_config"]["opponent"] =  tune.function(lambda x : (randrange(9), randrange(2)))

    return ppo.PPOTrainer(config = config, env = onevfixedgym.DuelFixedGym)

