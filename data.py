import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG, LOGGER  # 使用新的配置和日志对象

class ChatDataset(Dataset):
    """智能对话数据集，增强数据处理和错误恢复机制"""
    def __init__(self, data: list, tokenizer: Tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.skipped_samples = 0  # 使用实例变量记录跳过的样本
        
        # 验证特殊标记
        self._validate_special_tokens(tokenizer)
        
        # 处理数据，使用多线程加速
        self._process_data(data)
        
        if not self.data:
            raise ValueError("数据集为空，请检查数据源或调整处理参数")
        LOGGER.info(f"成功加载 {len(self.data)} 个样本，跳过 {self.skipped_samples} 个无效样本")

    def _validate_special_tokens(self, tokenizer: Tokenizer):
        """验证必要的特殊标记是否存在"""
        required_tokens = ["<|bos|>", "<|sep|>", "<|eos|>", "[PAD]"]
        missing_tokens = [token for token in required_tokens if tokenizer.token_to_id(token) is None]
        if missing_tokens:
            raise ValueError(f"缺少必要特殊标记: {', '.join(missing_tokens)}")
        LOGGER.debug("所有必要的特殊标记均已验证通过")

    def _process_data(self, data: list):
        """处理整个数据集，使用多线程并行处理""" 
        with ThreadPoolExecutor(max_workers=CONFIG.NUM_WORKER_THREADS) as executor:
            future_to_item = {executor.submit(self._safe_process_item, item): item for item in data}
            for future in tqdm(as_completed(future_to_item), total=len(data), desc="处理数据集"):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result:
                        self.data.append(result)
                    else:
                        self.skipped_samples += 1
                except Exception as e:
                    LOGGER.warning(f"处理样本失败: {str(e)}")
                    self.skipped_samples += 1

    def _safe_process_item(self, item: dict) -> dict:
        """安全地处理单个样本"""
        try:
            return self._process_item(item)
        except Exception as e:
            LOGGER.debug(f"样本处理异常: {str(e)}; 样本内容: {item}")
            return None

    def _process_item(self, item: dict) -> dict:
        """处理单个样本，并应用智能截断策略"""
        # 参数校验
        question = item.get('question', '').strip()
        answer = item.get('answer', '').strip()
        if not question or not answer:
            raise ValueError("问题或答案为空")
        
        # 构建输入文本
        prompt = f"<|bos|>{question}<|sep|>"
        full_text = f"{prompt}{answer}<|eos|>"
        
        # 编码文本
        encoding = self.tokenizer.encode(full_text)
        input_ids = encoding.ids
        
        # 生成标签（问题部分不计算损失）
        prompt_length = len(self.tokenizer.encode(prompt).ids)
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        # 智能截断处理
        return self._smart_truncate(input_ids, labels)

    def _smart_truncate(self, input_ids: list, labels: list) -> dict:
        """优先保留答案内容的动态截断策略"""
        sep_id = self.tokenizer.token_to_id("<|sep|>")
        try:
            sep_pos = input_ids.index(sep_id) + 1
        except ValueError:
            raise ValueError("样本中缺少必要的分隔符")
        
        max_len = CONFIG.MAX_LEN
        min_answer_len = CONFIG.MIN_ANSWER_LEN
        prompt_part = input_ids[:sep_pos]
        answer_part = input_ids[sep_pos:]
        labels_part = labels[sep_pos:]
        
        # 检查最小答案长度
        if len(answer_part) < min_answer_len:
            raise ValueError(f"答案长度不足: {len(answer_part)} < {min_answer_len}")
        
        # 动态截断逻辑
        if len(input_ids) > max_len:
            # 计算需要截断的长度
            truncate_len = len(input_ids) - max_len
            
            # 优先截断问题部分（保留至少50个token）
            min_question_len = 50
            if len(prompt_part) > min_question_len:
                actual_truncate = min(truncate_len, len(prompt_part) - min_question_len)
                prompt_part = prompt_part[actual_truncate:]
                truncate_len -= actual_truncate
            
            # 如果还需要截断，从答案尾部截断
            if truncate_len > 0:
                answer_part = answer_part[:-truncate_len]
                labels_part = labels_part[:-truncate_len]
        
        # 合并和填充
        final_input = prompt_part + answer_part
        final_labels = [-100]*len(prompt_part) + labels_part
        
        pad_len = max_len - len(final_input)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        final_input += [pad_id] * pad_len
        final_labels += [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(final_input, dtype=torch.long),
            "labels": torch.tensor(final_labels, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_dataloaders() -> tuple[DataLoader, DataLoader]:
    """准备训练和验证数据加载器"""
    # 加载原始数据
    if not CONFIG.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {CONFIG.RAW_DATA_PATH}")
    
    with CONFIG.RAW_DATA_PATH.open("r", encoding="utf-8") as f:
        try:
            full_data = json.load(f).get("data", [])
            if not full_data:
                raise ValueError("数据文件中没有有效的数据")
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON数据: {str(e)}")
    
    # 划分数据集
    val_size = max(1, int(len(full_data) * CONFIG.VAL_RATIO))
    train_data = full_data[:-val_size]
    val_data = full_data[-val_size:]
    LOGGER.info(f"训练集大小: {len(train_data)}，验证集大小: {len(val_data)}")
    
    # 加载分词器
    if not CONFIG.TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"分词器文件不存在: {CONFIG.TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(CONFIG.TOKENIZER_PATH))
    LOGGER.debug("分词器加载成功")
    
    # 创建数据集
    train_dataset = ChatDataset(train_data, tokenizer)
    val_dataset = ChatDataset(val_data, tokenizer)
    
    # 创建数据加载器
    loader_kwargs = {
        "batch_size": CONFIG.BATCH_SIZE,
        "pin_memory": True,
        "num_workers": CONFIG.NUM_WORKER_THREADS,
        "drop_last": True
    }
    
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    )