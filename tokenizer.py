try:
    import ujson as json  # 使用更快的 JSON 解析器
except ImportError:
    import json

from typing import List, Union
from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BpeTokenizer:
    # 预初始化不变的组件以减少每次实例化的开销
    DEFAULT_SPECIAL_TOKENS = [
        "[PAD]", "[UNK]", "<|bos|>", "<|eos|>", "<|sep|>",
        "[gMASK]", "[MASK]", "<sog>", "<eog>",
        "<|user|>", "<|system|>"
    ]
    NORMALIZER = normalizers.Sequence([normalizers.NFKC()])
    PRE_TOKENIZER = pre_tokenizers.Sequence([
        pre_tokenizers.Punctuation(),
        pre_tokenizers.Digits(individual_digits=True)
    ])
    DECODER = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
    POST_PROCESSOR = processors.ByteLevel(trim_offsets=True)

    def __init__(self, path: str = None, special_tokens: List[str] = None):
        self._special_tokens = special_tokens if special_tokens is not None else self.DEFAULT_SPECIAL_TOKENS

        if path is not None:
            logger.info(f"Loading tokenizer from file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)

            vocab = tokenizer_data.get('model', {}).get('vocab', {})
            merges = tokenizer_data.get('model', {}).get('merges', [])

            merges_tuples = [tuple(merge.split(' ')) for merge in merges if isinstance(merge, str) and len(merge.split(' ')) == 2]

            model = BPE(vocab=vocab, merges=merges_tuples, byte_fallback=True)
            tokenizer = Tokenizer(model)

            tokenizer.normalizer = self.NORMALIZER
            tokenizer.pre_tokenizer = self.PRE_TOKENIZER
            tokenizer.decoder = self.DECODER
            tokenizer.post_processor = self.POST_PROCESSOR

            tokenizer.add_special_tokens(self._special_tokens)
            self._tokenizer = tokenizer
            logger.info("Tokenizer loaded successfully.")
        else:
            logger.info("Initializing new BPE tokenizer...")
            model = BPE(byte_fallback=True)
            tokenizer = Tokenizer(model)

            tokenizer.normalizer = self.NORMALIZER
            tokenizer.pre_tokenizer = self.PRE_TOKENIZER
            tokenizer.decoder = self.DECODER
            tokenizer.post_processor = self.POST_PROCESSOR

            tokenizer.add_special_tokens(self._special_tokens)
            self._tokenizer = tokenizer
            logger.info("New BPE tokenizer initialized successfully.")

    def _init_trainer(self, vocab_size: int, min_freq: int) -> BpeTrainer:
        logger.info("Initializing BPE trainer...")
        alphabet = pre_tokenizers.ByteLevel().alphabet()
        min_size = len(self._special_tokens) + len(alphabet)
        if vocab_size < min_size:
            raise ValueError(f"vocab_size must be >= {min_size}, but got {vocab_size}")

        lim_len = vocab_size - len(self._special_tokens)
        trainer = BpeTrainer(
            initial_alphabet=alphabet,
            min_frequency=min_freq,
            vocab_size=lim_len,
            show_progress=True,
            special_tokens=self._special_tokens
        )
        logger.info("BPE trainer initialized successfully.")
        return trainer

    def encode(self, text: str, max_length: int = None, truncation: bool = False) -> List[int]:
        """
        增强编码逻辑，支持截断
        """
        encoded: Encoding = self._tokenizer.encode(text)

        if truncation and max_length is not None:
            if max_length < 1:
                raise ValueError("max_length 必须大于0")
            encoded = encoded.truncate(max_length)

        return encoded.ids

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    def train(self, files: List[str], vocab_size: int, min_freq: int):
        logger.info(f"Training tokenizer with files: {files}")
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train(files=files, trainer=trainer)
        logger.info("Tokenizer training completed successfully.")

    def save(self, path: str):
        try:
            logger.info(f"Saving tokenizer to file: {path}")
            self._tokenizer.save(path)
            logger.info("Tokenizer saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")

    def load(self, path: str):
        try:
            logger.info(f"Loading tokenizer from file: {path}")
            self._tokenizer = Tokenizer.from_file(path)
            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()