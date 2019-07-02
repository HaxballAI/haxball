from gym.envs.registration import register

register(
    id='haxball-v0',
    entry_point='gym_haxball.envs:HaxballEnv',
)
