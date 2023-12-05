## Structured State Space Models for In-Context Reinforcement Learning

#### At NeurIPS 2023

Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob Foerster, Satinder Singh, Feryal Behbahani

This is a [PureJAX](https://github.com/luchris429/purejaxrl) version of our NeurIPS 2023 paper "Structured State Space Models for In-Context Reinforcement Learning". We evaluate and modify S4-like models for reinforcement learning. Furthermore, we re-implemented [POPGym](https://arxiv.org/abs/2303.01859) in pure JAX, speeding up future research in partially-observed RL.

If you use this repository, please cite:

```
@article{lu2023structured,
  title={Structured State Space Models for In-Context Reinforcement Learning},
  author={Lu, Chris and Schroecker, Yannick and Gu, Albert and Parisotto, Emilio and Foerster, Jakob and Singh, Satinder and Behbahani, Feryal},
  journal={arXiv preprint arXiv:2303.03982},
  year={2023}
}
```

## Installation

Install dependencies using the requirements.txt file:

```
pip install -r requirements.txt
```

In order to use JAX on your accelerators, you can find more details in the [JAX documentation](https://github.com/google/jax#installation).

```
pip install "jax[cuda12_pip]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Example Usage

`python3 run_popgym.py --num-runs 1 --env BattleshipEasy --arch s5`

## TODOs:

I expect to complete these by 4 November.

- Releasing more thorough tests for each environment

- More thorough benchmarks for speedups

## References and Acknowledgments

The code implementations here are heavily inspired by:

- [POPGym](https://github.com/proroklab/popgym)
- [S5](https://github.com/lindermanlab/S5/tree/main)
- [Gymnax](https://github.com/RobertTLange/gymnax)
- [PureJaxRL](https://github.com/luchris429/purejaxrl/tree/main)

If you use this repository, please cite:

```
@article{lu2023structured,
  title={Structured State Space Models for In-Context Reinforcement Learning},
  author={Lu, Chris and Schroecker, Yannick and Gu, Albert and Parisotto, Emilio and Foerster, Jakob and Singh, Satinder and Behbahani, Feryal},
  journal={arXiv preprint arXiv:2303.03982},
  year={2023}
}
```

If you use the relevant components from above, please also cite them. This includes:

S5
```
@inproceedings{
smith2023simplified,
title={Simplified State Space Layers for Sequence Modeling},
author={Jimmy T.H. Smith and Andrew Warrington and Scott Linderman},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Ai8Hw3AXqks}
}
```

POPGym
```
@inproceedings{
morad2023popgym,
title={{POPG}ym: Benchmarking Partially Observable Reinforcement Learning},
author={Steven Morad and Ryan Kortvelesy and Matteo Bettini and Stephan Liwicki and Amanda Prorok},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=chDrutUTs0K}
}
```

Gymnax
```
@software{gymnax2022github,
  author = {Robert Tjarko Lange},
  title = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url = {http://github.com/RobertTLange/gymnax},
  version = {0.0.4},
  year = {2022},
}
```

PureJaxRL
```
@article{lu2022discovered,
    title={Discovered policy optimisation},
    author={Lu, Chris and Kuba, Jakub and Letcher, Alistair and Metz, Luke and Schroeder de Witt, Christian and Foerster, Jakob},
    journal={Advances in Neural Information Processing Systems},
    volume={35},
    pages={16455--16468},
    year={2022}
}
```