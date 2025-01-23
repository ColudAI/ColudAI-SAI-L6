from tokenizers import Tokenizer, Encoding
from tokenizers import decoders, processors, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class BpeTokenizer:
    def __init__(self, path=None):
        """
        初始化BpeTokenizer
        :param path: 已保存的分词器文件的路径。如果提供，则从该文件中加载分词器
        """
        self._special_tokens = [
            "[gMASK]", "[MASK]", "<sog>", "<eog>",
            "<|user|>", "<|system|>"
        ]

        if path is None:
            model = BPE(byte_fallback=True)
            tokenizer = Tokenizer(model)

            tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Punctuation(),
                pre_tokenizers.Digits(individual_digits=True)
            ])

            tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False, use_regex=True)
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

            self._tokenizer = tokenizer
        else:
            self._tokenizer = Tokenizer.from_file(path)

    def _init_trainer(self, vocab_size, min_freq):
        """
        使用给定的词汇表大小和最小频率初始化BpeTrainer.

        :param vocab_size: 词汇表的目标大小.
        :param min_freq: 一个标记被包含在词汇表中的最小频率。
        :return: BpeTrainer的一个实例。
        """
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        special_tokens = self._special_tokens  # 获取特殊 token 列表
        min_size = len(special_tokens) + len(alphabet)
        assert vocab_size >= min_size, f"vocab_size must be greater than or equal to {min_size}"

        lim_len = vocab_size - len(special_tokens)
        trainer = BpeTrainer(
            initial_alphabet=alphabet,
            min_frequency=min_freq,
            vocab_size=lim_len,
            show_progress=True,
            special_tokens=special_tokens  # 在 Trainer 中指定特殊 token
        )
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
        # 注意：这里不需要再次添加特殊 token，因为已经在 Trainer 中指定了

    def train_from_iterator(self, iterator, vocab_size, min_freq):
        """
        Train the tokenizer from an iterator.

        :param iterator: Iterator over training data.
        :param vocab_size: Target size of the vocabulary.
        :param min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        trainer = self._init_trainer(vocab_size, min_freq)
        self._tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
        # 注意：这里不需要再次添加特殊 token，因为已经在 Trainer 中指定了

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
        将一个标记转换为其对应的标记ID

        :param token: 标记.
        :return: 对应的标记ID。
        """
        return self._tokenizer.token_to_id(token)

    def __len__(self) -> int:
        """
        获取词汇表的大小

        :return: 词汇表大小。
        """
        return self._tokenizer.get_vocab_size()  # 使用 get_vocab_size() 方法