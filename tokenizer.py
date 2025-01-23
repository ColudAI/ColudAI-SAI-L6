from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class BpeTokenizer:
    def __init__(self, path=None):
        """
        Initialize the BpeTokenizer.

        :param path: Path to a saved tokenizer file. If provided, loads the tokenizer from this file.
        """
        self._special_tokens = [
            "[gMASK]", "[MASK]", "<sog>", "<eog>",
            "<|user|>", "<|system|>"
        ]

        model = BPE(byte_fallback=True)
        tokenizer = Tokenizer(model)

        tokenizer.normalizer = normalizers.Sequence()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Punctuation(),
            pre_tokenizers.Digits(individual_digits=True)
        ])

        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        if path is not None:
            tokenizer = Tokenizer.from_file(path)

        self._tokenizer = tokenizer

    def _init_trainer(self, vocab_size, min_freq):
        """
        Initialize the BpeTrainer with given vocabulary size and minimum frequency.

        :param vocab_size: The target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        :return: An instance of BpeTrainer.
        """
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size = len(self._special_tokens) + len(alphabet)
        assert vocab_size >= min_size, f"vocab_size must be greater than or equal to {min_size}"

        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer(special_tokens=self._special_tokens, vocab_size=vocab_size, min_frequency=min_freq)
        return trainer

    def train(self, files, vocab_size, min_freq):
        """
        Train the tokenizer on a list of files.

        :param files: List of paths to training files.
        :param vocab_size: Target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train(files=files, trainer=trainer)

    def train_from_iterator(self, iterator, vocab_size, min_freq):
        """
        Train the tokenizer from an iterator.

        :param iterator: Iterator over training data.
        :param vocab_size: Target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)

    def save(self, path):
        """
        Save the tokenizer to a file.

        :param path: Path where the tokenizer will be saved.
        """
        self._tokenizer.save(path)

    def load(self, path):
        """
        Load the tokenizer from a file.

        :param path: Path to the saved tokenizer file.
        """
        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str, out_type: type = int) -> list:
        """
        Encode a string into tokens.

        :param text: Input text to encode.
        :param out_type: Type of output (int or str).
        :return: List of token IDs or tokens.
        """
        assert out_type in [int, str], f"out_type must be int or str, but got {out_type}"
        encoded: Encoding = self._tokenizer.encode(text)
        if out_type == int:
            return encoded.ids
        else:
            return encoded.tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        :param tokens: List of token IDs.
        :return: Decoded string.
        """
        return self._tokenizer.decode(tokens)

    def id_to_token(self, token_id: int) -> str:
        """
        Convert a token ID to its corresponding token.

        :param token_id: Token ID.
        :return: Corresponding token.
        """
        return self._tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its corresponding token ID.

        :param token: Token.
        :return: Corresponding token ID.
        """
        return self._tokenizer.token_to_id(token)

    def __len__(self) -> int:
        """
        Get the size of the vocabulary.

        :return: Vocabulary size.
        """
        return self._tokenizer.get_vocab_size()