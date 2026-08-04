from src.data import download_data
from src.tokenizer import Tokenizer
from src.pack import pack_data

if __name__ == "__main__":
    # download_data()
    tokenizer = Tokenizer.train(overwrite=False)
    print(
        f"tokenizer ready: vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id}"
    )
    pack_data(tokenizer)
 
