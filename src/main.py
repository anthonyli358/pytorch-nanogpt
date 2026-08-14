import torch
import math

from src.config import GPTConfig
from src.data.data import download_data
from src.models.tokenizer import Tokenizer
from src.data.pack import pack_data
from src.models.gpt import GPT

if __name__ == "__main__":
    # --- Download Data ---
    # download_data()  # already downloaded

    # --- Prepare Tokenizer ---
    tokenizer = Tokenizer.train(overwrite=False)
    print(
        f"tokenizer ready: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id}"
    )
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
