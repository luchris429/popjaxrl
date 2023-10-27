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
    value_cards: jnp.ndarray
    query_cards: jnp.ndarray
    running_count: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

class CountRecall(environment.Environment):
    def __init__(self, num_decks=1, num_types=2):
        super().__init__()
        self.decksize = 52
        # self.error_clamp = error_clamp # NOT IMPLEMENTED
        self.num_decks = num_decks
        self.num_types = num_types
        self.num_cards = self.decksize * self.num_decks
        self.max_num = self.num_cards // self.num_types
        self.reward_scale = 1.0 / self.num_cards
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        running_count = state.running_count.at[state.value_cards[state.timestep]].add(1)
        prev_count = state.running_count[state.query_cards[state.timestep]]

        reward = jnp.where(action==prev_count, self.reward_scale, -self.reward_scale)
        new_state = EnvState(state.timestep + 1, state.value_cards, state.query_cards, running_count)
        obs = self.get_obs(new_state)
        terminated = new_state.timestep == self.num_cards

        return obs, new_state, reward, terminated, {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_value, key_query = jax.random.split(key, 3)
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_types
        value_cards = jax.random.permutation(key_value, cards)
        query_cards = jax.random.permutation(key_query, cards)
        running_count = jnp.zeros((self.num_types,))
        state = EnvState(
            timestep=0,
            value_cards=value_cards,
            query_cards=query_cards,
            running_count=running_count,
        )
        # obs = state.cards[state.timestep]
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs = jnp.zeros((2 * self.num_types))
        obs = obs.at[state.value_cards[state.timestep]].set(1)
        obs = obs.at[self.num_types + state.query_cards[state.timestep]].set(1)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((2 * self.num_types,)), jnp.ones((2 * self.num_types,)), (2 * self.num_types,), dtype=jnp.float32)

class CountRecallEasy(CountRecall):
    def __init__(self):
        super().__init__(num_decks=1, num_types=2)

class CountRecallMedium(CountRecall):
    def __init__(self):
        super().__init__(num_decks=2, num_types=4)

class CountRecallHard(CountRecall):
    def __init__(self):
        super().__init__(num_decks=4, num_types=13)
