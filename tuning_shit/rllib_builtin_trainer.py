import ray
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.ppo as ppo

from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog
from ray.tune.logger import pretty_print

from gym_haxball.singleplayergym import SingleplayerGym


def main():
    ray.init()
    
    '''config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1
    trainer = ppo.PPOTrainer(config=config, env=SingleplayerGym)'''

    
    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 2
    config["num_workers"] = 6
    trainer = a3c.A3CTrainer(config=config, env=SingleplayerGym)

    # Can optionally call trainer.restore(path) to load a checkpoint.

    for i in range(1000):
       # Perform one iteration of training the policy with PPO
       result = trainer.train()
       print(pretty_print(result))

       if i % 100 == 0:
           checkpoint = trainer.save()
           print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    main()