import math

import torch

from src.config import GPTConfig
from src.data.data import download_data
from src.data.pack import pack_data
from src.eval.winrate import run_eval
from src.inference.generate import generate
from src.inference.speculative import compare_decoding
from src.make_preferences import make_preferences
from src.models.gpt import GPT
from src.models.tokenizer import Tokenizer
from src.training.pretrain import train as pretrain
from src.training.sft import train_sft
from src.training.dpo import train_dpo
from src.training.grpo import train_grpo
from src.training.ppo import train_ppo
from src.training.draft import train_draft


if __name__ == "__main__":
    # --- Download Data ---
    # download_data()  # already downloaded

    # --- Prepare Tokenizer ---
    tokenizer = Tokenizer.train(overwrite=False)
    print(
        f"tokenizer ready: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id}"
    )

    # --- Pack Data ---
    pack_data(tokenizer)

    # --- Initialise a model, untrained ---
    torch.manual_seed(0)
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"total params        : {model.num_params(non_embedding=False):,}")
    print(f"non-embedding params: {model.num_params():,}")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    _, loss = model(x, y)
    print(f"init loss: {loss.item():.4f}  (expect ~ ln(vocab) = {math.log(cfg.vocab_size):.4f})")

    model.eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=8, top_k=50)
    print("generate out shape:", tuple(out.shape))

    # --- Pretrain the base model ---
    # pretrain()  # long run; checkpoints under checkpoints/

    # --- Sample from a trained checkpoint ---
    # generate()

    # --- SFT (instruction following) ---
    # train_sft()

    # --- Preference pairs (feeds DPO / RL) ---
    # make_preferences()

    # --- DPO ---
    # train_dpo()

    # --- PPO / GRPO (verifiable-reward RL) ---
    # train_ppo()
    # train_grpo()

    # --- Draft model for speculative decoding ---
    # train_draft()

    # --- Evaluate models ---
    run_eval()
    compare_decoding()  # speculative decoding
