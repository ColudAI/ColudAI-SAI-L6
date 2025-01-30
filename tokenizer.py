import json 
from typing import List, Union 
from tokenizers import Tokenizer, Encoding 
from tokenizers import decoders, processors, normalizers, pre_tokenizers 
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer 
 
class BpeTokenizer:
    def __init__(self, path: str = None, special_tokens: List[str] = None):
        """
        Initialize the BpeTokenizer.
 
        :param path: Path to a saved tokenizer file. If provided, loads the tokenizer from this file.
        :param special_tokens: List of special tokens to add to the tokenizer.
        """
        self._special_tokens = [
            "[PAD]", "[UNK]", "<|bos|>", "<|eos|>", "<|sep|>",
            "[gMASK]", "[MASK]", "<sog>", "<eog>",
            "<|user|>", "<|system|>"
        ] if special_tokens is None else special_tokens 
 
        if path is not None:
            print(f"Loading tokenizer from file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
 
            vocab = tokenizer_data['model']['vocab']
            merges = tokenizer_data['model']['merges']
            merges_tuples = [tuple(merge.split(' ')) for merge in merges if isinstance(merge, str) and len(merge.split(' ')) == 2]
 
            model = BPE(vocab=vocab, merges=merges_tuples, byte_fallback=True)
            tokenizer = Tokenizer(model)
 
            tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Punctuation(),
                pre_tokenizers.Digits(individual_digits=True)
            ])
 
            tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
 
            tokenizer.add_special_tokens(self._special_tokens)
            self._tokenizer = tokenizer 
            print("Tokenizer loaded successfully.")
        else:
            print("Initializing new BPE tokenizer...")
            model = BPE(byte_fallback=True)
            tokenizer = Tokenizer(model)
 
            tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Punctuation(),
                pre_tokenizers.Digits(individual_digits=True)
            ])
 
            tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
 
            tokenizer.add_special_tokens(self._special_tokens)
            self._tokenizer = tokenizer 
            print("New BPE tokenizer initialized successfully.")
 
    def _init_trainer(self, vocab_size: int, min_freq: int) -> BpeTrainer:
        """
        Initialize the BpeTrainer with given vocabulary size and minimum frequency.
 
        :param vocab_size: The target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        :return: An instance of BpeTrainer.
        """
        print("Initializing BPE trainer...")
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        min_size = len(self._special_tokens) + len(alphabet)
        assert vocab_size >= min_size, f"vocab_size must be greater than or equal to {min_size}"
 
        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer(
            initial_alphabet=alphabet,
            min_frequency=min_freq,
            vocab_size=lim_len,
            show_progress=True,
            special_tokens=self._special_tokens 
        )
        print("BPE trainer initialized successfully.")
        return trainer 
 
    def train(self, files: List[str], vocab_size: int, min_freq: int):
        """
        Train the tokenizer on a list of files.
 
        :param files: List of paths to training files.
        :param vocab_size: Target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        print(f"Training tokenizer with files: {files}")
        print(f"Target vocab size: {vocab_size}, Min frequency: {min_freq}")
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train(files=files, trainer=trainer)
        print("Tokenizer training completed successfully.")
 
    def train_from_iterator(self, iterator, vocab_size: int, min_freq: int):
        """
        Train the tokenizer from an iterator.
 
        :param iterator: Iterator over training data.
        :param vocab_size: Target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        print("Training tokenizer from iterator...")
        print(f"Target vocab size: {vocab_size}, Min frequency: {min_freq}")
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        print("Tokenizer training from iterator completed successfully.")
 
    def save(self, path: str):
        """
        Save the tokenizer to a file.
 
        :param path: Path where the tokenizer will be saved.
        """
        print(f"Saving tokenizer to file: {path}")
        self._tokenizer.save(path)
        print("Tokenizer saved successfully.")
 
    def load(self, path: str):
        """
        Load the tokenizer from a file.
 
        :param path: Path to the saved tokenizer file.
        """
        print(f"Loading tokenizer from file: {path}")
        self._tokenizer = Tokenizer.from_file(path)
        print("Tokenizer loaded successfully.")
 
    def encode(self, text: str, out_type: type = int) -> Union[List[int], List[str]]:
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
 
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.
 
        :param tokens: List of token IDs.
        :return: Decoded string.
        """
        return self._tokenizer.decode(tokens)
 
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