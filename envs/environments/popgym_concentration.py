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
    face_up: jnp.ndarray
    in_play: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

class Concentration(environment.Environment):
    def __init__(self, num_decks=1, num_types=2):
        super().__init__()
        self.decksize = 52
        # self.error_clamp = error_clamp # NOT IMPLEMENTED (?)
        self.num_decks = num_decks
        self.num_types = num_types
        self.num_cards = self.decksize * self.num_decks
        self.episode_length = jnp.ceil(2 * self.num_cards - (self.num_cards / (2.0 * self.num_cards - 1)))
        self.success_reward_scale = 1.0 / (self.num_cards // 2)
        self.failure_reward_scale = -1.0 / (self.episode_length)
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        reward = 0.0
        flipped_same_idx = state.in_play[action] == 1
        new_in_play = state.in_play.at[action].set(1)
        state_obs = state.replace(in_play=new_in_play)
        obs = self.get_obs(state_obs)
        trying_card_already_up = jnp.any((state.face_up * new_in_play) > 0.0)
        valid_play = jnp.logical_and(
            1 - trying_card_already_up,
            jnp.logical_or(flipped_same_idx, jnp.sum(new_in_play) == 2),
        )

        # IF TRYING CARD ALREADY UP
        reward = jnp.where(trying_card_already_up, 
            jnp.sum(new_in_play) * self.failure_reward_scale, # WHY IS IT SCALED BY NUM IN PLAY?
            0.0
        )
        new_in_play = jnp.where(trying_card_already_up, jnp.zeros_like(new_in_play), new_in_play)

        # ELIF VALID PLAY
        # import pdb; pdb.set_trace()
        in_play_idx = jnp.nonzero(new_in_play, size=2, fill_value=0)[0]
        cards_match = state.cards[in_play_idx[0]] == state.cards[in_play_idx[1]]
        # flipped_same_idx = in_play_idx[0] == in_play_idx[1]
        cards_match_and_not_same = cards_match * (1 - flipped_same_idx)
        reward = jnp.where( # IF SUCCESSFUL
            cards_match_and_not_same * valid_play, 
            self.success_reward_scale,
            reward
        )
        new_face_up = jnp.where(
            cards_match_and_not_same * valid_play,
            state.face_up.at[in_play_idx].set(1),
            state.face_up,
        )
        terminated = new_face_up.sum() == self.num_cards
        reward = jnp.where( # IF FLIPPED SAME IDX OR NOT MATCHED
            (1 - cards_match_and_not_same) * valid_play, 
            2 * self.failure_reward_scale,
            reward,
        )
        new_in_play = jnp.where(
            valid_play,
            jnp.zeros_like(new_in_play),
            new_in_play,
        )
        new_state = EnvState(
            state.timestep + 1,
            state.cards,
            new_face_up,
            new_in_play,
        )
        terminated = jnp.logical_or(terminated, new_state.timestep == self.episode_length)

        return obs, new_state, reward, terminated, {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        key, key_value = jax.random.split(key)
        cards = jnp.arange(self.decksize * self.num_decks) % self.num_types
        cards = jax.random.permutation(key_value, cards)
        state = EnvState(
            timestep=0,
            cards=cards,
            face_up=jnp.zeros_like(cards),
            in_play=jnp.zeros_like(cards),
        )
        # obs = state.cards[state.timestep]
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state."""
        obs_enc = jnp.ones((self.num_cards,)) * self.num_types
        visible = jnp.logical_or(state.face_up, state.in_play)
        obs_enc = jnp.where(visible, state.cards, obs_enc)
        one_hot = jax.nn.one_hot(obs_enc, self.num_types+1)
        obs = jnp.ravel(one_hot)
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_cards)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((self.num_cards * (self.num_types+1),)), 
                          jnp.ones((self.num_cards * (self.num_types+1),)), 
                          (self.num_cards * (self.num_types+1),), dtype=jnp.float32)

class ConcentrationEasy(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, num_types=2)

class ConcentrationMedium(Concentration):
    def __init__(self):
        super().__init__(num_decks=2, num_types=2)

class ConcentrationHard(Concentration):
    def __init__(self):
        super().__init__(num_decks=1, num_types=13)