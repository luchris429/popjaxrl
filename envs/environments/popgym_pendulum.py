import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

def angle_normalize(x: float) -> float:
    """Normalize the angle - radians."""
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

@struct.dataclass
class EnvState:
    theta: float
    theta_dot: float
    last_u: float  # Only needed for rendering
    time: int

@struct.dataclass
class EnvParams:
    max_speed: float = 8.0
    max_torque: float = 2.0
    dt: float = 0.05
    g: float = 10.0  # gravity
    m: float = 1.0  # mass
    l: float = 1.0  # length

class NoisyStatelessPendulum(environment.Environment):
    """
    JAX Compatible version of Pendulum-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """

    def __init__(self, noise_sigma=0.0, max_steps_in_episode=200):
        super().__init__()
        self.obs_shape = (2,)
        self.noise_sigma = noise_sigma
        self.max_steps_in_episode = max_steps_in_episode

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Integrate pendulum ODE and return transition."""
        u = jnp.clip(action, -params.max_torque, params.max_torque)
        reward = -(
            angle_normalize(state.theta) ** 2
            + 0.1 * state.theta_dot ** 2
            + 0.001 * (u ** 2)
        )
        reward = reward.squeeze()
        reward = self.reward_transform(params, reward)

        newthdot = state.theta_dot + (
            (
                3 * params.g / (2 * params.l) * jnp.sin(state.theta)
                + 3.0 / (params.m * params.l ** 2) * u
            )
            * params.dt
        )

        newthdot = jnp.clip(newthdot, -params.max_speed, params.max_speed)
        newth = state.theta + newthdot * params.dt

        # Update state dict and evaluate termination conditions
        state = EnvState(
            newth.squeeze(), newthdot.squeeze(), u.reshape(), state.time + 1
        )
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(key, state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reward_transform(self, params: EnvParams, reward: float):
        low, high = -16.2736044, 0
        shifted = reward + (high - low) / 2
        scaled = shifted / ((high - low) / 2)
        transformed = scaled / self.max_steps_in_episode
        return transformed

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling theta, theta_dot."""
        high = jnp.array([jnp.pi, 1])
        state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
        state = EnvState(theta=state[0], theta_dot=state[1], last_u=0.0, time=0)
        key, _ = jax.random.split(key)
        return self.get_obs(key, state, params), state

    def get_obs(self, key: chex.PRNGKey, state: EnvState, params: EnvParams) -> chex.Array:
        """Return angle in polar coordinates and change."""
        return jnp.array(
            [
                jnp.cos(state.theta),
                jnp.sin(state.theta),
                # state.theta_dot,
            ]
        ).squeeze() + jax.random.normal(key, shape=(2,)) * self.noise_sigma

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done = state.time >= self.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=-params.max_torque,
            high=params.max_torque,
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([1.0, 1.0], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(2,), dtype=jnp.float32)
        # return spaces.Box(-high, high, shape=(3,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "theta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "theta_dot": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "last_u": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "time": spaces.Discrete(self.max_steps_in_episode),
            }
        )

class NoisyStatelessPendulumEasy(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.1, max_steps_in_episode=200)

class NoisyStatelessPendulumMedium(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.2, max_steps_in_episode=200)

class NoisyStatelessPendulumHard(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.3, max_steps_in_episode=200)

class StatelessPendulumEasy(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=200)

class StatelessPendulumMedium(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=150)

class StatelessPendulumHard(NoisyStatelessPendulum):
    def __init__(self):
        super().__init__(noise_sigma=0.0, max_steps_in_episode=100)