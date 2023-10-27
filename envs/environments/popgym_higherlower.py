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
    cards: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

class HigherLower(environment.Environment):
    def __init__(self, num_decks=1):
        super().__init__()
        self.num_ranks = 13
        self.decksize = 52
        self.num_decks = num_decks
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        num_cards = self.decksize * self.num_decks
        reward_scale = 1.0 / (num_cards)

        guess_higher = action == 0
        curr_value = state.cards[state.timestep]
        next_value = state.cards[state.timestep + 1]
        reward = jnp.where(guess_higher == (next_value > curr_value), reward_scale, -reward_scale)
        reward = jnp.where(next_value == curr_value, 0, reward)

        new_state = EnvState(state.timestep + 1, state.cards)
        terminated = new_state.timestep == num_cards
        obs = self.get_obs(new_state)

        return obs, new_state, reward, terminated, {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_ranks
        cards = jax.random.permutation(key, cards)
        state = EnvState(
            timestep=0,
            cards=cards,
        )
        # obs = state.cards[state.timestep]
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs = jnp.zeros((self.num_ranks,))
        obs = obs.at[state.cards[state.timestep]].set(1)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((self.num_ranks,)), jnp.ones((self.num_ranks,)), (self.num_ranks,), dtype=jnp.float32)

class HigherLowerEasy(HigherLower):
    def __init__(self):
        super().__init__(num_decks=1)

class HigherLowerMedium(HigherLower):
    def __init__(self):
        super().__init__(num_decks=2)

class HigherLowerHard(HigherLower):
    def __init__(self):
        super().__init__(num_decks=3)
