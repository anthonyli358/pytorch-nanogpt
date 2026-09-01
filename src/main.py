from src.data.data import download_data
from src.data.pack import pack_data
from src.eval.winrate import run_eval
from src.inference.generate import generate
from src.inference.speculative import compare_decoding
from src.data.make_preferences import make_preferences
from src.models.gpt import test_gpt
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
    # tokenizer = Tokenizer.train(overwrite=False)

    # --- Pack Data ---
    # pack_data(tokenizer)

    # --- Initialise a model, untrained (sanity check) ---
    # test_gpt()

    # --- Model training ---

    # pretrain()  # long run; checkpoints under checkpoints/
    # generate()  # sample from a trained checkpoint
    # train_sft()
    # make_preferences()  # preference pairs, feeds DPO
    # train_dpo()
    # train_ppo()
    # train_grpo()
    # train_draft()  # draft model for speculative decoding

    # --- Evaluate models ---
    run_eval()
    compare_decoding()  # speculative decoding
