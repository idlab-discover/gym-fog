from gym.envs.registration import register

register(
    id='FogEnvEnergyEfficiencyLarge-v0',
    entry_point='gym_fog.envs:FogEnvEnergyEfficiencyLarge',
)

register(
    id='FogEnvEnergyEfficiencySmall-v0',
    entry_point='gym_fog.envs:FogEnvEnergyEfficiencySmall',
)