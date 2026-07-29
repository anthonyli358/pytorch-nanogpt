from src.data import download_data
from src.tokenizer import Tokenizer

if __name__ == "__main__":
    # download_data()
    tokenizer = Tokenizer.train(overwrite=True)
    print(
        f"tokenizer ready: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id}"
    )
