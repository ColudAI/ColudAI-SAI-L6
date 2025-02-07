import os
import sys
import logging
from pathlib import Path
from typing import Optional
import torch
from datetime import datetime

class Config:
    """全局配置类（单例模式）"""
    
    # ==================== 路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # ==================== 硬件配置 ====================
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    USE_AMP: bool = True  # 自动混合精度
    USE_BF16: bool = False  # 需要A100+GPU
    TF32_PRECISION: bool = True  # 启用TF32
    NUM_WORKER_THREADS: int = min(os.cpu_count(), 8)  # 数据加载线程数
    
    # ==================== 训练参数 ====================
    EPOCHS: int = 10
    BATCH_SIZE: int = 32
    GRAD_ACCUM: int = 4  # 梯度累积步数
    BASE_LR: float = 2e-5
    WEIGHT_DECAY: float = 0.01  # 权重衰减系数
    WARMUP_STEPS: int = 4000
    CLIP_GRAD_NORM: float = 1.0  # 梯度裁剪阈值
    BALANCE_COEF: float = 0.01  # MoE平衡系数
    DROPOUT: float = 0.1
    LABEL_SMOOTHING: float = 0.1  # 标签平滑系数
    
    # ==================== 模型架构 ====================
    VOCAB_SIZE: int = 512 # 不要轻易修改，否则无法运行
    EMBED_SIZE: int = 256 # 不要轻易修改，否则无法运行
    NUM_HEADS: int = 8
    HIDDEN_DIM: int = 1024
    ENC_LAYERS: int = 6
    DEC_LAYERS: int = 6
    USE_MOE: bool = True  # 是否使用混合专家
    NUM_EXPERTS: int = 8  # 专家数量
    EXPERT_DIM: int = 512  # 专家层维度
    DATALOADER_NUM_WORKERS: int = 4
    NUM_WORKER_THREADS: int = 8
    TOKENIZER_VOCAB_SIZE: int = 30000
    MAX_LEN = 512
    MIN_ANSWER_LEN = 1
    # ==================== 数据参数 ====================
    VAL_RATIO: float = 0.1  # 验证集比例
    GENERATION_MAX_LENGTH: int = 512  # 生成最大长度
    GENERATION_TEMPERATURE: float = 0.9  # 生成温度
    GENERATION_TOP_K: int = 50  # Top-k采样
    GENERATION_TOP_P: float = 0.95  # Top-p采样
    
    # ==================== 特殊标记 ====================
    BOS_TOKEN: str = "<|bos|>"
    EOS_TOKEN: str = "<|eos|>"
    SEP_TOKEN: str = "<|sep|>"
    PAD_TOKEN: str = "[PAD]"

    #==================== 调试模式 ======================
    DEBUG: bool = True #这啥用没有，我也不知道为什么要写，不写会报错....
    
    def __init__(self):
        """初始化必要目录和日志系统"""
        self._init_paths()
        self._init_logger()
        self._set_torch_precision()
        
    def _init_paths(self):
        """创建必要目录并设置路径常量"""
        # 创建基础目录
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 设置路径常量
        self.RAW_DATA_PATH: Path = self.DATA_DIR / "train_data.json"
        self.TOKENIZER_PATH: Path = self.DATA_DIR / "tokenizer.json"
        self.MODEL_SAVE_PATH: Path = self.MODEL_DIR / "model.pth"
        
    def _init_logger(self):
        """配置多处理器日志系统"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.LOG_DIR / f"training_{timestamp}.log"
        
        # 日志格式配置
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # 文件处理器（记录所有级别）
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器（仅记录INFO以上级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # 主日志器配置
        self.logger = logging.getLogger("Main")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 抑制第三方库日志
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("tokenizers").setLevel(logging.WARNING)
        
    def _set_torch_precision(self):
        """配置PyTorch计算精度"""
        if self.TF32_PRECISION and torch.cuda.is_available():
            # 启用TF32矩阵加速（Ampere+架构）
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
    def __getattr__(self, name):
        """安全属性访问保护"""
        raise AttributeError(f"'Config' object has no attribute '{name}'")

# 单例配置实例
CONFIG = Config()

# 导出日志器别名
LOGGER = CONFIG.logger

if __name__ == "__main__":
    # 配置验证测试
    LOGGER.info("=== 配置验证测试 ===")
    LOGGER.info(f"计算设备: {CONFIG.DEVICE}")
    LOGGER.info(f"混合精度模式: {'Enabled' if CONFIG.USE_AMP else 'Disabled'}")
    LOGGER.info(f"总训练周期: {CONFIG.EPOCHS}")
    LOGGER.info(f"梯度累积步数: {CONFIG.GRAD_ACCUM}")
    LOGGER.info(f"MoE专家数量: {CONFIG.NUM_EXPERTS}")
    LOGGER.info("=== 路径验证 ===")
    LOGGER.info(f"模型保存路径: {CONFIG.MODEL_SAVE_PATH}")
    LOGGER.info(f"原始数据路径: {CONFIG.RAW_DATA_PATH.exists()}")
    LOGGER.info("配置验证通过！")