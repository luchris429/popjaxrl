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

class Autoencode(environment.Environment):
    def __init__(self, num_decks=1):
        super().__init__()
        self.num_suits = 4
        self.decksize = 52
        self.num_decks = num_decks
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        num_cards = self.decksize * self.num_decks
        reward = 0

        reward_scale = 1.0 / (num_cards)

        terminated = state.timestep == num_cards * 2
        play = state.timestep >= num_cards

        reward = jnp.where(
            # jnp.logical_and(play, state.cards[-(state.timestep - num_cards - 1)] == action),
            jnp.flip(state.cards, axis=0)[state.timestep - num_cards] == action,
            reward_scale,
            -reward_scale,
        )
        reward = jnp.where(
            play,
            reward,
            0,
        )

        new_state = EnvState(state.timestep + 1, state.cards)
        obs = self.get_obs(new_state)

        return obs, new_state, reward, terminated, {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_suits
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
        play = state.timestep >= self.decksize * self.num_decks
        play_obs = jnp.zeros((self.num_suits,))
        watch_obs = play_obs.at[state.cards[state.timestep]].set(1)
        obs =  jnp.where(play, watch_obs, play_obs)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_suits)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((self.num_suits,)), jnp.ones((self.num_suits,)), (self.num_suits,), dtype=jnp.float32)

class AutoencodeEasy(Autoencode):
    def __init__(self):
        super().__init__(num_decks=1)

class AutoencodeMedium(Autoencode):
    def __init__(self):
        super().__init__(num_decks=2)

class AutoencodeHard(Autoencode):
    def __init__(self):
        super().__init__(num_decks=3)