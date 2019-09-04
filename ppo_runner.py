import basic_trainers.ppo_fixed as ppo
import torch
import gym_haxball.solofixedgym as solo

if __name__ == "__main__":
    mod1 = torch.load("models/seb_mod_baseline.model")
    mod2 = torch.load("models/seb_mod_baseline.model")
    mod1.to("cuda")
    mod2.to("cpu")
    trainer = ppo.PPOTrainer(mod1, solo.env_constructor(mod2), 10, parallelise = True)
    trainer.train(100)
    torch.save(mod1, "models/seb_test.model")
