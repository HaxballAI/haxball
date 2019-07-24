import ray
from ray import tune
import os

from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog

from gym_haxball.singleplayergym import SingleplayerGym


def main():
    # Run "tensorboard --logdir=rllib_saved_models/basic-v2" to visualise
    ray.init()

    save_local_path = "rllib_saved_models/basic-v2/"
    save_abs_path = os.path.join(os.getcwd(), save_local_path)

    tune.run(
        "A2C",
        stop={
            "timesteps_total": 3000000,
        },
        config={
            "env": SingleplayerGym,
            #"model": {
            #    "custom_model": "my_model",
            #},
            "lr": 0.001,#tune.grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 5,  # parallelism
            "num_gpus": 0,
            "env_config": {
                "step_length": 7,
                "max_steps": 400
            },
        },
        checkpoint_freq=1000,
        local_dir=save_abs_path,
        checkpoint_at_end=True
    )


if __name__ == "__main__":
    main()
