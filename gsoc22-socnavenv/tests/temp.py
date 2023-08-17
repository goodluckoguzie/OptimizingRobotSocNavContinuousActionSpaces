import socnavenv
import gym

env = gym.make("SocNavEnv-v1")
env.configure(".../configs/env_timestep_0_25.yaml")
env.set_padded_observations(True)



obs, _ = env.reset()


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        env.reset()