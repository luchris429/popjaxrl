import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import GymnaxWrapper, environment, Optional, partial, Tuple, chex, spaces, Union
from flax import struct

class AliasPrevAction(GymnaxWrapper):
    """Adds a t0 flag and the last action."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        action_space = self._env.action_space(params)
        og_observation_space = self._env.observation_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([og_observation_space.low, jnp.array([0.0, 0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.n - 1, 1.0])])
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([og_observation_space.low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.high]), jnp.array([1.0])])
        else:
            raise NotImplementedError
        return spaces.Box(
            low=low,
            high=high,
            shape=(self._env.observation_space(params).shape[-1]+2,), # NOTE: ASSUMES FLAT RIGHT NOW
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.concatenate([obs, jnp.array([0.0, 1.0])])
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.array([action, 0.0])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([0.0])])
        return obs, state, reward, done, info

class AliasPrevActionV2(GymnaxWrapper):
    """Adds a t0 flag and the last action."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        action_space = self._env.action_space(params)
        og_observation_space = self._env.observation_space(params)
        if type(action_space) == spaces.Discrete:
            low = jnp.concatenate([og_observation_space.low, jnp.zeros((action_space.n+1,))])
            high = jnp.concatenate([og_observation_space.high, jnp.ones((action_space.n+1,))])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+action_space.n+1,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        elif type(action_space) == spaces.Box:
            low = jnp.concatenate([og_observation_space.low, jnp.array([action_space.low]), jnp.array([0.0])])
            high = jnp.concatenate([og_observation_space.high, jnp.array([action_space.high]), jnp.array([1.0])])
            return spaces.Box(
                low=low,
                high=high,
                shape=(self._env.observation_space(params).shape[-1]+2,), # NOTE: ASSUMES FLAT RIGHT NOW
                dtype=self._env.observation_space(params).dtype,
            )
        else:
            raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        action_space = self._env.action_space(params)
        obs, state = self._env.reset(key, params)
        if isinstance(action_space, spaces.Box):
            obs = jnp.concatenate([obs, jnp.array([0.0, 1.0])])
        elif isinstance(action_space, spaces.Discrete):
            obs = jnp.concatenate([obs, jnp.zeros((action_space.n,)), jnp.array([1.0])])
        else:
            raise NotImplementedError
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_space = self._env.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            # obs = jnp.concatenate([obs, jnp.array([action, 0.0])])
            action_in = jnp.zeros((action_space.n,))
            action_in = action_in.at[action].set(1.0)
            obs = jnp.concatenate([obs, action_in, jnp.array([0.0])])
        else:
            obs = jnp.concatenate([obs, action, jnp.array([0.0])])
        return obs, state, reward, done, info

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state = env_state,
            episode_returns = new_episode_return * (1 - done),
            episode_lengths = new_episode_length * (1 - done),
            returned_episode_returns = state.returned_episode_returns * (1 - done) + new_episode_return * done,
            returned_episode_lengths = state.returned_episode_lengths * (1 - done) + new_episode_length * done,
            timestep = state.timestep + 1,
        )
        # info["returned_episode_returns"] = state.returned_episode_returns
        # info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        info["return_info"] = jnp.stack([state.timestep, state.returned_episode_returns])
        # info["timestep"] = state.timestep
        return obs, state, reward, done, info
