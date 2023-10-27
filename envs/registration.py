from .environments import (
    NoisyStatelessMetaCartPole,
    AutoencodeEasy,
    AutoencodeMedium,
    AutoencodeHard,
    BattleshipEasy,
    BattleshipMedium,
    BattleshipHard,
    StatelessCartPoleEasy,
    StatelessCartPoleMedium,
    StatelessCartPoleHard,
    NoisyStatelessCartPoleEasy,
    NoisyStatelessCartPoleMedium,
    NoisyStatelessCartPoleHard,
    ConcentrationEasy,
    ConcentrationMedium,
    ConcentrationHard,
    CountRecallEasy,
    CountRecallMedium,
    CountRecallHard,
    HigherLowerEasy,
    HigherLowerMedium,
    HigherLowerHard,
    MineSweeperEasy,
    MineSweeperMedium,
    MineSweeperHard,
    MultiarmedBanditEasy,
    MultiarmedBanditMedium,
    MultiarmedBanditHard,
    StatelessPendulumEasy,
    StatelessPendulumMedium,
    StatelessPendulumHard,
    NoisyStatelessPendulumEasy,
    NoisyStatelessPendulumMedium,
    NoisyStatelessPendulumHard,
    RepeatFirstEasy,
    RepeatFirstMedium,
    RepeatFirstHard,
    RepeatPreviousEasy,
    RepeatPreviousMedium,
    RepeatPreviousHard,
)

def make(env_id: str, **env_kwargs):
    if env_id == "NoisyStatelessMetaCartPole":
        env = NoisyStatelessMetaCartPole(**env_kwargs)
    elif env_id == "AutoencodeEasy":
        env = AutoencodeEasy(**env_kwargs)
    elif env_id == "AutoencodeMedium":
        env = AutoencodeMedium(**env_kwargs)
    elif env_id == "AutoencodeHard":
        env = AutoencodeHard(**env_kwargs)
    elif env_id == "BattleshipEasy":
        env = BattleshipEasy(**env_kwargs)
    elif env_id == "BattleshipMedium":
        env = BattleshipMedium(**env_kwargs)
    elif env_id == "BattleshipHard":
        env = BattleshipHard(**env_kwargs)
    elif env_id == "StatelessCartPoleEasy":
        env = StatelessCartPoleEasy(**env_kwargs)
    elif env_id == "StatelessCartPoleMedium":
        env = StatelessCartPoleMedium(**env_kwargs)
    elif env_id == "StatelessCartPoleHard":
        env = StatelessCartPoleHard(**env_kwargs)
    elif env_id == "NoisyStatelessCartPoleEasy":
        env = NoisyStatelessCartPoleEasy(**env_kwargs)
    elif env_id == "NoisyStatelessCartPoleMedium":
        env = NoisyStatelessCartPoleMedium(**env_kwargs)
    elif env_id == "NoisyStatelessCartPoleHard":
        env = NoisyStatelessCartPoleHard(**env_kwargs)
    elif env_id == "ConcentrationEasy":
        env = ConcentrationEasy(**env_kwargs)
    elif env_id == "ConcentrationMedium":
        env = ConcentrationMedium(**env_kwargs)
    elif env_id == "ConcentrationHard":
        env = ConcentrationHard(**env_kwargs)
    elif env_id == "CountRecallEasy":
        env = CountRecallEasy(**env_kwargs)
    elif env_id == "CountRecallMedium":
        env = CountRecallMedium(**env_kwargs)
    elif env_id == "CountRecallHard":
        env = CountRecallHard(**env_kwargs)
    elif env_id == "HigherLowerEasy":
        env = HigherLowerEasy(**env_kwargs)
    elif env_id == "HigherLowerMedium":
        env = HigherLowerMedium(**env_kwargs)
    elif env_id == "HigherLowerHard":
        env = HigherLowerHard(**env_kwargs)
    elif env_id == "MinesweeperEasy":
        env = MineSweeperEasy(**env_kwargs)
    elif env_id == "MinesweeperMedium":
        env = MineSweeperMedium(**env_kwargs)
    elif env_id == "MinesweeperHard":
        env = MineSweeperHard(**env_kwargs)
    elif env_id == "MultiArmedBanditEasy":
        env = MultiarmedBanditEasy(**env_kwargs)
    elif env_id == "MultiArmedBanditMedium":
        env = MultiarmedBanditMedium(**env_kwargs)
    elif env_id == "MultiArmedBanditHard":
        env = MultiarmedBanditHard(**env_kwargs)
    elif env_id == "StatelessPendulumEasy":
        env = StatelessPendulumEasy(**env_kwargs)
    elif env_id == "StatelessPendulumMedium":
        env = StatelessPendulumMedium(**env_kwargs)
    elif env_id == "StatelessPendulumHard":
        env = StatelessPendulumHard(**env_kwargs)
    elif env_id == "NoisyStatelessPendulumEasy":
        env = NoisyStatelessPendulumEasy(**env_kwargs)
    elif env_id == "NoisyStatelessPendulumMedium":
        env = NoisyStatelessPendulumMedium(**env_kwargs)
    elif env_id == "NoisyStatelessPendulumHard":
        env = NoisyStatelessPendulumHard(**env_kwargs)
    elif env_id == "RepeatFirstEasy":
        env = RepeatFirstEasy(**env_kwargs)
    elif env_id == "RepeatFirstMedium":
        env = RepeatFirstMedium(**env_kwargs)
    elif env_id == "RepeatFirstHard":
        env = RepeatFirstHard(**env_kwargs)
    elif env_id == "RepeatPreviousEasy":
        env = RepeatPreviousEasy(**env_kwargs)
    elif env_id == "RepeatPreviousMedium":
        env = RepeatPreviousMedium(**env_kwargs)
    elif env_id == "RepeatPreviousHard":
        env = RepeatPreviousHard(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params
