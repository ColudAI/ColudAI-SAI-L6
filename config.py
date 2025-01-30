import torch
import logging
from pathlib import Path

class BaseConfig:
    """基础配置类"""
    # 路径配置
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # 硬件配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available()
    FP16_PRECISION = "high"

class TrainingConfig(BaseConfig):
    """训练专用配置"""
    # 路径配置
    TOKENIZER_PATH = BaseConfig.DATA_DIR / "tokenizer.json"
    MODEL_SAVE_PATH = BaseConfig.MODELS_DIR / "transformer.pth"
    RAW_DATA_PATH = BaseConfig.DATA_DIR / "train_data.json"
    
    # 模型参数
    VOCAB_SIZE = 30000
    MAX_LEN = 512
    EMBED_SIZE = 256
    NUM_HEADS = 8
    HIDDEN_DIM = 1024
    ENC_LAYERS = 8
    DEC_LAYERS = 8
    
    # 训练参数
    EPOCHS = 16
    BATCH_SIZE = 32
    GRAD_ACCUM = 4
    BASE_LR = 2e-5
    WARMUP_STEPS = 4000
    DROPOUT = 0.2
    LABEL_SMOOTHING = 0.1
    CLIP_GRAD_NORM = 1.0
    
    # MoE参数
    USE_MOE = True
    NUM_EXPERTS = 4
    EXPERT_DIM = 512
    BALANCE_COEF = 0.01
    
    # 数据参数
    MIN_ANSWER_LEN = 32
    VAL_RATIO = 0.1

class ExistingConfig(BaseConfig):
    """兼容旧配置"""
    # 旧路径配置
    DATA_PATH = BaseConfig.DATA_DIR / "legacy_data.json"
    MODEL_PATH = BaseConfig.MODELS_DIR / "legacy_model.pth"
    
    # 旧训练参数
    EPOCHS = 10
    BATCH_SIZE = 16
    BASE_LR = 1e-4

class GenerationConfig:
    # 生成参数
    GENERATION_MAX_LENGTH = 512        # 最大生成长度
    GENERATION_TEMPERATURE = 0.9       # 温度参数 (0.1~1.5)
    GENERATION_TOP_K = 50              # Top-K采样参数 (0表示禁用)
    GENERATION_TOP_P = 0.95            # Top-P采样参数 (1.0表示禁用)
    
    # 特殊标记
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    SEP_TOKEN = "<|sep|>"
    PAD_TOKEN = "[PAD]"

class RuntimeConfig(TrainingConfig, ExistingConfig):
    """
    运行时实际使用的配置
    继承顺序说明：
    - TrainingConfig 在后，会覆盖 ExistingConfig 中的同名属性
    - BaseConfig 作为基类提供基础配置
    """
    def __init__(self):
        # 自动创建必要目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging():
    """日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# 初始化配置和日志
CONFIG = RuntimeConfig()
LOGGER = setup_logging()

if __name__ == "__main__":
    # 配置验证
    LOGGER.info(f"当前设备: {CONFIG.DEVICE}")
    LOGGER.info(f"训练数据路径: {CONFIG.RAW_DATA_PATH}")
    LOGGER.info(f"模型保存路径: {CONFIG.MODEL_SAVE_PATH}")
    LOGGER.info(f"是否启用MoE: {CONFIG.USE_MOE}")