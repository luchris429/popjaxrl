import jax.numpy as jnp
import jax
import flax.linen as nn
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional
from flax.linen.initializers import constant, orthogonal
import numpy as np
from .popgym_cartpole import NoisyStatelessCartPole, EnvParams, EnvState


class MetaAugNetwork(nn.Module):
    out_size: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.out_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x

@struct.dataclass
class MetaEnvState:
    obs_params: chex.Array
    trial_num: int
    total_steps: int
    env_state: EnvState
    init_state: Optional[chex.Array]
    init_obs: Optional[chex.Array]

@struct.dataclass
class MetaEnvParams:
    num_trials_per_episode: int = 16
    env_params: EnvParams = EnvParams()

class NoisyStatelessMetaCartPole(environment.Environment):

    def __init__(self):
        super().__init__()
        self.env = NoisyStatelessCartPole(max_steps_in_episode=200, noise_sigma=0.0)
        self.obs_shape = (7,)
        self.obs_aug = MetaAugNetwork(4)

    @property
    def default_params(self) -> MetaEnvParams:
        return MetaEnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: MetaEnvState, action: int, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)

        env_obs_st, env_state_st, reward, env_done, info = self.env.step_env(key, state.env_state, action, params.env_params)
        # env_obs_re, env_state_re = self.env.reset_env(key_reset, params.env_params)
        env_obs_re, env_state_re = state.init_obs, state.init_state

        env_state = jax.tree_map(
            lambda x, y: jax.lax.select(env_done, x, y), env_state_re, env_state_st
        )
        env_obs = jax.lax.select(env_done, env_obs_re, env_obs_st)
        env_obs = self.obs_aug.apply(state.obs_params, env_obs[None,:])[0]

        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvState(
            obs_params=state.obs_params,
            trial_num=trial_num,
            total_steps=total_steps,
            env_state=env_state,
            init_state=state.init_state,
            init_obs=state.init_obs,
        )

        obs = jnp.concatenate([env_obs, jnp.array([action, env_done, 0.0])])

        return (
            obs,
            state,
            reward,
            done,
            info,
        )
    
    def reset_env(
        self, key: chex.PRNGKey, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        env_key, obs_key = jax.random.split(key)
        env_obs, env_state = self.env.reset_env(env_key, params.env_params)
        obs_params = self.obs_aug.init(obs_key, env_obs[None,:])
        test = self.obs_aug.apply(obs_params, env_obs[None,:])[0]
        state = MetaEnvState(
            obs_params=obs_params,
            trial_num=0,
            total_steps=0,
            env_state=env_state,
            init_state = env_state,
            init_obs = env_obs,
        )
        obs = jnp.concatenate([test, jnp.array([0.0, 0.0, 1.0])])

        return obs, state

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[MetaEnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.ones([4+3])
        return spaces.Box(-high, high, (7,), dtype=jnp.float32)
