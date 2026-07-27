# pytorch-nanogpt
Pytorch implementation of a GPT style decoder

## Usage

To use CUDA 12.6 for torch, we add the pytorch index as [described in the documentation](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index) to [pyproject.toml](pyproject.toml). 

## Development

1. Download the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) from Huggingface and [store them](src/data.py).

