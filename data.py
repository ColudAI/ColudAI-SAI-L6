import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG, LOGGER
from typing import Tuple, List, Optional
import random

class ChatDataset(Dataset):
    """智能对话数据集，增强数据处理和错误恢复机制"""

    def __init__(self, data: List[dict], tokenizer: Tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.skipped_samples = 0

        self._validate_special_tokens()
        self._process_data(data)

        if not self.data:
            raise ValueError("数据集为空，请检查数据源或调整处理参数")
        LOGGER.info(f"成功加载 {len(self.data)} 个样本，跳过 {self.skipped_samples} 个无效样本")

    def _validate_special_tokens(self):
        required_tokens = ["<|bos|>", "<|sep|>", "<|eos|>", "[PAD]"]
        missing_tokens = [token for token in required_tokens if self.tokenizer.token_to_id(token) is None]
        if missing_tokens:
            raise ValueError(f"缺少必要特殊标记: {', '.join(missing_tokens)}")
        LOGGER.debug("所有必要的特殊标记均已验证通过")

    def _process_data(self, data: List[dict]):
        with ThreadPoolExecutor(max_workers=CONFIG.NUM_WORKER_THREADS) as executor:
            futures = {executor.submit(self._safe_process_item, item): item for item in data}
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理数据集"):
                try:
                    result = future.result()
                    if result:
                        self.data.append(result)
                    else:
                        self.skipped_samples += 1
                except Exception as e:
                    LOGGER.warning(f"处理样本失败: {str(e)}")
                    self.skipped_samples += 1

    def _safe_process_item(self, item: dict) -> Optional[dict]:
        try:
            return self._process_item(item)
        except ValueError as ve:
            LOGGER.warning(f"值错误: {str(ve)}; 样本内容: {item}")
        except Exception as e:
            LOGGER.warning(f"处理样本失败: {str(e)}; 样本内容: {item}")
        return None

    def _process_item(self, item: dict) -> dict:
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()

        if not question or not answer:
            raise ValueError(f"无效样本: 问题或答案为空 -> {item}")

        full_text = f"<|bos|>{question}<|sep|>{answer}<|eos|>"  # 去除多余空格

        # 增加编码参数校验
        try:
            encoding = self.tokenizer.encode(
                full_text,
                add_special_tokens=False,
                #truncation=False
            )
        except Exception as e:
            LOGGER.error(f"编码失败: {full_text} | 错误: {str(e)}")
            raise

        if not encoding.ids:
            raise ValueError(f"编码结果为空: {full_text}")

        # 使用tokenizer内置方法获取特殊token
        sep_id = self.tokenizer.token_to_id("<|sep|>")
        eos_id = self.tokenizer.token_to_id("<|eos|>")
        
        # 验证必要token存在性
        if None in [sep_id, eos_id]:
            missing = ["<|sep|>" if sep_id is None else "", "<|eos|>" if eos_id is None else ""]
            raise ValueError(f"缺少必要特殊标记: {', '.join(filter(None, missing))}")

        # 重构截断逻辑
        return self._smart_truncate(encoding.ids)

    def _smart_truncate(self, input_ids: List[int]) -> dict:
        max_len = CONFIG.MAX_LEN
        min_answer_len = CONFIG.MIN_ANSWER_LEN
        pad_id = self.tokenizer.token_to_id("[PAD]")
        sep_id = self.tokenizer.token_to_id("<|sep|>")
        eos_id = self.tokenizer.token_to_id("<|eos|>")

        try:
            sep_pos = input_ids.index(sep_id)
        except ValueError:
            raise ValueError(f"未找到分隔符: {input_ids}")

        # 分离prompt和answer
        prompt_part = input_ids[:sep_pos+1]  # 包含sep标记
        answer_part = input_ids[sep_pos+1:]

        # 答案长度验证
        if len(answer_part) < min_answer_len:
            raise ValueError(f"答案过短: {len(answer_part)} < {min_answer_len}")

        # 智能截断策略
        total_length = len(prompt_part) + len(answer_part)
        if total_length > max_len:
            overflow = total_length - max_len
            # 保留完整答案，截断prompt
            keep_prompt_len = max(50, len(prompt_part) - overflow)
            prompt_part = prompt_part[-keep_prompt_len:]
            answer_part = answer_part[:max_len - keep_prompt_len]

        # 合并并生成labels
        final_input = prompt_part + answer_part
        labels = [-100]*len(prompt_part) + answer_part  # 仅计算answer部分的loss

        # 填充处理
        if len(final_input) < max_len:
            pad_len = max_len - len(final_input)
            final_input += [pad_id] * pad_len
            labels += [-100] * pad_len
        else:
            final_input = final_input[:max_len]
            labels = labels[:max_len]

        return {
            "input_ids": torch.tensor(final_input, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


class DataLoaderPreparer:
    """改进后的数据加载器准备类"""

    def __init__(self):
        self.data_path = Path(CONFIG.RAW_DATA_PATH)
        self.batch_size = CONFIG.BATCH_SIZE
        self.val_ratio = CONFIG.VAL_RATIO
        self.tokenizer_path = Path(CONFIG.TOKENIZER_PATH)
        self.num_workers = CONFIG.DATALOADER_NUM_WORKERS
        self._validate_inputs()  # 现在该方法存在

    def _validate_inputs(self):
        """参数校验增强版"""
        # 检查数据文件
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        if self.data_path.suffix != ".json":
            raise ValueError("仅支持JSON格式数据文件")
        
        # 检查批次参数
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size必须是正整数，当前值: {self.batch_size}")
        
        # 验证集比例检查
        if not (0 < self.val_ratio < 1):
            raise ValueError(f"val_ratio必须在0和1之间，当前值: {self.val_ratio}")
        
        # 工作进程数检查
        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError(f"num_workers必须是非负整数，当前值: {self.num_workers}")
        
        # 分词器路径检查
        if self.tokenizer_path.exists() and not self.tokenizer_path.is_file():
            raise ValueError(f"分词器路径必须是文件: {self.tokenizer_path}")

    def prepare(self) -> Tuple[DataLoader, DataLoader]:
        # 加载原始数据
        with open(self.data_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)["data"]

        # 加载/训练tokenizer
        tokenizer = self._load_or_train_tokenizer(full_data)

        # 拆分数据集
        random.shuffle(full_data)
        split_idx = int(len(full_data) * (1 - self.val_ratio))
        train_data, val_data = full_data[:split_idx], full_data[split_idx:]

        # 创建数据集
        train_dataset = ChatDataset(train_data, tokenizer)
        val_dataset = ChatDataset(val_data, tokenizer)

        # 数据加载器配置
        loader_kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": True,
            "num_workers": self.num_workers,
            "drop_last": True,
            "persistent_workers": self.num_workers > 0
        }

        return (
            DataLoader(train_dataset, shuffle=True, **loader_kwargs),
            DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        )

    def _load_or_train_tokenizer(self, data: List[dict]) -> Tokenizer:
        if not self.tokenizer_path.exists():
            self._train_tokenizer(data)
            LOGGER.info(f"新分词器已保存至: {self.tokenizer_path}")
        return Tokenizer.from_file(str(self.tokenizer_path))

    def _train_tokenizer(self, data: List[dict]):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # 特殊标记配置
        special_tokens = ["<|bos|>", "<|sep|>", "<|eos|>", "[PAD]", "[UNK]"]
        tokenizer.add_special_tokens(special_tokens)

        # 预分词配置
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation()
        ])

        # 后处理配置
        post_processor = processors.TemplateProcessing(
            single="<|bos|> $A <|eos|>",
            special_tokens=[("<|bos|>", tokenizer.token_to_id("<|bos|>")),
                            ("<|eos|>", tokenizer.token_to_id("<|eos|>"))]
        )
        tokenizer.post_processor = post_processor

        # 准备训练语料
        corpus = [
            f"<|bos|>{item['question']}<|sep|>{item['answer']}<|eos|>"
            for item in data
            if item.get('question') and item.get('answer')
        ]

        # 训练参数
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG.TOKENIZER_VOCAB_SIZE,
            special_tokens=special_tokens,
            min_frequency=2,
            continuing_subword_prefix="##"
        )

        # 开始训练
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        tokenizer.save(str(self.tokenizer_path))