import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

@struct.dataclass
class EnvState:
    timestep: int
    payouts: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

class MultiarmedBandit(environment.Environment):
    def __init__(self, num_bandits=10, episode_length=200):
        super().__init__()
        self.num_bandits = num_bandits
        self.episode_length = episode_length
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        key, key_sample = jax.random.split(key)
        obs = jnp.where(jax.random.uniform(key_sample, (1,)) < state.payouts[action], 
                        jnp.array([1.0]), jnp.array([0.0]))
        reward = jnp.where(obs, 1.0 / self.episode_length, -1.0 / self.episode_length)[0]
        new_state = EnvState(timestep=state.timestep + 1, payouts=state.payouts)
        terminated = new_state.timestep >= self.episode_length

        return obs, new_state, reward, terminated, {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        payouts = jax.random.uniform(key, (self.num_bandits,))
        state = EnvState(
            timestep=0,
            payouts=payouts,
        )

        return jnp.array([0.0]), state
    
    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_bandits)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((1,)), jnp.ones((1,)), (1,), dtype=jnp.float32)

class MultiarmedBanditEasy(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=10, episode_length=200)

class MultiarmedBanditMedium(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=20, episode_length=400)

class MultiarmedBanditHard(MultiarmedBandit):
    def __init__(self):
        super().__init__(num_bandits=30, episode_length=600)