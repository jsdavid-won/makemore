# makemore

Character-level language models trained on a dataset of 32K names, following Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series.

## Notebooks

| Notebook | Topic | Key Concepts |
|----------|-------|-------------|
| `build_makemore_part1.ipynb` | Bigram models | Count-based bigrams, single linear layer, negative log-likelihood |
| `build_makemore_part2.ipynb` | MLP | Character embeddings, hidden layers, train/val split, learning rate finder |
| `build_makemore_part3.ipynb` | Internals | Batch normalization, weight initialization (Kaiming), deep MLPs, custom PyTorch-style modules |
| `build_makemore_part4.ipynb` | Backprop | Manual backward pass through cross-entropy, batchnorm, and the full network |
| `build_makemore_part5.ipynb` | WaveNet | Hierarchical model with `FlattenConsec` (dilated convolutions), `Embedding`, `Sequential` |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Data

`names.txt` contains ~32K human names used for training, validation, and test splits (80/10/10).
