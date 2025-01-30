import math
import logging
import torch
from pathlib import Path
from typing import Dict, Union, Optional
from config import TrainingConfig, logger

def count_parameters(model: torch.nn.Module) -> str:
    """统计模型参数数量（包含可训练/不可训练参数）
    
    Args:
        model (torch.nn.Module): 待分析的PyTorch模型
        
    Returns:
        str: 格式化的参数字符串，示例："总参数: 150M (150,000,000) | 可训练参数: 120M"
        
    Raises:
        TypeError: 当输入不是PyTorch模块时抛出异常
    """
    if not isinstance(model, torch.nn.Module):
        logger.error("输入必须是torch.nn.Module类型")
        raise TypeError("输入必须是torch.nn.Module类型")
    
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 自动选择单位 (B/M/K)
        def format_num(num: int) -> str:
            if num >= 1e9:
                return f"{num/1e9:.1f}B"
            elif num >= 1e6:
                return f"{num/1e6:.1f}M"
            elif num >= 1e3:
                return f"{num/1e3:.1f}K"
            return str(num)
            
        return (f"总参数: {format_num(total_params)} ({total_params:,}) | "
                f"可训练参数: {format_num(trainable_params)}")
                
    except Exception as e:
        logger.error(f"参数统计失败: {str(e)}")
        raise

def load_checkpoint(model: torch.nn.Module, 
                   checkpoint_path: Union[str, Path],
                   strict: bool = True) -> None:
    """加载模型检查点（兼容单GPU/DDP模式）
    
    Args:
        model (torch.nn.Module): 要加载参数的模型
        checkpoint_path (Union[str, Path]): 检查点文件路径
        strict (bool): 是否严格匹配参数名称，默认为True
        
    Raises:
        FileNotFoundError: 当检查点文件不存在时抛出
        RuntimeError: 当参数加载失败时抛出
    """
    try:
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")
            
        # 自动选择设备
        device = TrainingConfig().DEVICE
        state_dict = torch.load(path, map_location=device)
        
        # 处理DDP模型前缀
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            
        # 加载状态字典
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        if missing:
            logger.warning(f"缺失参数: {missing}")
        if unexpected:
            logger.warning(f"意外参数: {unexpected}")
            
        logger.info(f"成功从 {path} 加载检查点")
        
    except Exception as e:
        logger.error(f"加载检查点失败: {str(e)}")
        raise RuntimeError(f"加载检查点失败: {str(e)}") from e

def save_checkpoint(model: torch.nn.Module, 
                   save_path: Union[str, Path],
                   overwrite: bool = False) -> None:
    """保存模型检查点（兼容DDP模式）
    
    Args:
        model (torch.nn.Module): 要保存的模型
        save_path (Union[str, Path]): 保存路径
        overwrite (bool): 是否覆盖已存在文件，默认为False
        
    Raises:
        FileExistsError: 当文件已存在且overwrite为False时抛出
        RuntimeError: 保存过程中发生错误时抛出
    """
    try:
        path = Path(save_path).resolve()
        if path.exists() and not overwrite:
            raise FileExistsError(f"文件已存在: {path} (使用overwrite=True强制覆盖)")
            
        # 创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理DDP模型
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        torch.save(state_dict, path)
        logger.info(f"模型成功保存至: {path}")
        
    except Exception as e:
        logger.error(f"保存检查点失败: {str(e)}")
        raise RuntimeError(f"保存检查点失败: {str(e)}") from e

def set_seed(seed: Optional[int] = None) -> None:
    """设置全局随机种子（包含PyTorch/CUDA/Numpy）
    
    Args:
        seed (int, optional): 随机种子值。如果为None，则从配置读取默认值
    """
    try:
        cfg = TrainingConfig()
        if seed is None:
            seed = cfg.SEED  # 假设config.py中定义了SEED参数
            
        import numpy as np
        import random
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"全局随机种子设置为: {seed} (deterministic={cfg.DETERMINISTIC})")
        
    except Exception as e:
        logger.error(f"设置随机种子失败: {str(e)}")
        raise

def get_hardware_utilization() -> Dict[str, str]:
    """获取当前硬件资源使用情况
    
    Returns:
        Dict[str, str]: 包含以下键值的字典:
            - "cpu_usage": CPU使用率百分比
            - "gpu_memory": GPU显存使用情况 (如果可用)
            - "gpu_utilization": GPU计算利用率 (如果可用)
    """
    stats = {}
    
    try:
        import psutil
        # CPU使用率
        stats["cpu_usage"] = f"{psutil.cpu_percent()}%"
    except ImportError:
        stats["cpu_usage"] = "N/A (需要安装psutil)"
    
    # GPU信息
    if torch.cuda.is_available():
        try:
            device = TrainingConfig().DEVICE
            props = torch.cuda.get_device_properties(device)
            
            # 显存使用
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            total = props.total_memory / 1024**3
            stats["gpu_memory"] = f"{allocated:.2f}/{total:.2f} GB"
            
            # 计算利用率（兼容不同版本）
            if hasattr(torch.cuda, "utilization"):
                stats["gpu_utilization"] = f"{torch.cuda.utilization(device)}%"
            else:
                stats["gpu_utilization"] = "N/A (需要PyTorch 1.7+)"
                
        except Exception as e:
            stats["gpu"] = f"获取GPU信息失败: {str(e)}"
    else:
        stats["gpu"] = "N/A"
        
    return stats

def enable_tf32_precision() -> None:
    """启用TF32矩阵运算（适用于Ampere+架构GPU）"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("已启用TF32矩阵运算")
    else:
        logger.warning("TF32仅在NVIDIA GPU上可用")

def model_size_in_mb(model: torch.nn.Module) -> float:
    """计算模型参数的存储大小（以MB为单位）
    
    Args:
        model (torch.nn.Module): 待分析的模型
        
    Returns:
        float: 模型参数的存储大小（MB）
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return round(size_mb, 2)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """生成因果掩码"""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask