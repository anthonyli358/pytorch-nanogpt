import os
from pathlib import Path
import sentencepiece as spm

from src.config import (
    DATA_DIR,
    FILE_SETS,
    EOS_MARKER,
    VOCAB_SIZE,
)

TOKENIZER_DIR = "data/tokenizer"
MODEL_PREFIX = "spm"  # data/tokenizer/spm.model + spm.vocab
RAW_TRAIN = Path(DATA_DIR) / FILE_SETS["train"]
MODEL_PATH = Path(TOKENIZER_DIR) / f"{MODEL_PREFIX}.model"


class Tokenizer:
    """Thin wrapper around a trained SentencePiece model.

    Attributes:
        eos_id: Id of the <|endoftext|> piece (document / EOS boundary).
        pad_id: Id of the <pad> piece.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        """Load a trained SentencePiece model.

        Args:
            model_path: Path to the .model file produced by
                :func:`train_tokenizer`.
        """
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.eos_id = self.sp.piece_to_id(EOS_MARKER)
        self.pad_id = self.sp.pad_id()

    @classmethod  # train and build instance at init
    def train(
        cls,
        input_path: Path = RAW_TRAIN,
        tokenizer_dir: Path = Path(TOKENIZER_DIR),
        vocab_size: int = VOCAB_SIZE,
        character_coverage: float = 1.0,
        input_sentence_size: int = 2_000_000,
        max_sentence_length: int = 8192,
        overwrite: bool = False,
    ) -> "Tokenizer":
        """Train a BPE model on the raw train text (or reuse a cached one) and load it.

        If a trained model already exists it is loaded as-is unless
        overwrite is set, so this is safe to call unconditionally in a
        pipeline. Lines are subsampled and shuffled, so training does not read
        the full ~2GB corpus. `<|endoftext|>` is kept whole via
        user_defined_symbols.

        Args:
            input_path: Path to the raw train .txt file (stories separated by
                <|endoftext|> lines).
            tokenizer_dir: Directory to write spm.model and spm.vocab into.
            vocab_size: Target vocabulary size, including specials and the 256
                byte-fallback pieces.
            character_coverage: Fraction of characters the model must cover; 1.0
                for clean English (byte_fallback covers the rest).
            input_sentence_size: Subsample this many lines for training rather
                than reading the full corpus.
            max_sentence_length: Byte cap per training line; longer lines are
                skipped.
            overwrite: If True, retrain even when a cached model exists.

        Returns:
            A loaded :class:`Tokenizer` backed by the trained model.
        """
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = tokenizer_dir / MODEL_PREFIX
        model_path = model_prefix.with_suffix(".model")

        if model_path.exists() and not overwrite:
            print(f"tokenizer already trained: {model_path}")
            return cls(model_path)

        spm.SentencePieceTrainer.train(
            input=str(input_path),
            model_prefix=str(model_prefix),
            model_type="bpe",
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            byte_fallback=True,  # never emit <unk>; fall back to bytes
            user_defined_symbols=[EOS_MARKER],  # keep <|endoftext|> as one piece
            unk_id=0,
            pad_id=1,
            bos_id=-1,  # decoder-only LM, corpus is a text stream
            eos_id=-1,  # disable native BOS/EOS; use EOS_MARKER
            unk_piece="<unk>",
            pad_piece="<pad>",
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            max_sentence_length=max_sentence_length,
            num_threads=os.cpu_count() or 4,
        )
        tok = cls(model_path)
        print(f"tokenizer ready: vocab_size={tok.vocab_size}, eos_id={tok.eos_id}")
        return tok

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text to token ids.

        Args:
            text: Input string.
            add_eos: If True, append the <|endoftext|> id. Appending the id
                directly is intentional -- encoding the literal marker string
                would pick up a leading add_dummy_prefix whitespace token.

        Returns:
            List of token ids.
        """
        ids = self.sp.encode(text, out_type=int)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text.

        Args:
            ids: List of token ids.

        Returns:
            The reconstructed string.
        """
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Number of pieces in the vocabulary."""
        return self.sp.vocab_size()
