import math
import logging
import torch
from pathlib import Path
from typing import Dict, Union, Optional, Tuple, List
from config import CONFIG, LOGGER  # 使用新的配置和日志对象

def count_parameters(model: torch.nn.Module) -> Tuple[str, str]:
    """增强版模型参数统计（返回可读字符串和原始数值）
    
    Args:
        model (torch.nn.Module): 待分析的PyTorch模型
        
    Returns:
        tuple: (格式化字符串, 原始数据字典)
        
    Raises:
        TypeError: 输入类型错误时抛出
    """
    if not isinstance(model, torch.nn.Module):
        LOGGER.error("输入类型错误，应为torch.nn.Module")
        raise TypeError("输入必须是torch.nn.Module类型")
    
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 智能单位转换
        def human_format(num: int) -> str:
            for unit in ["", "K", "M", "B"]:
                if abs(num) < 1000:
                    return f"{num:.1f}{unit}"
                num /= 1000
            return f"{num:.1f}T"
            
        # 生成统计信息
        stats_str = (
            f"总参数: {human_format(total)} ({total:,}) | "
            f"可训练参数: {human_format(trainable)} ({trainable:,})"
        )
        stats_raw = {"total": total, "trainable": trainable}
        
        return stats_str, stats_raw
        
    except Exception as e:
        LOGGER.error(f"参数统计失败: {str(e)}", exc_info=True)
        raise

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    strict: bool = True,
    map_location: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """增强版模型加载（支持混合精度/DDP/多设备）
    
    Args:
        model: 目标模型
        checkpoint_path: 检查点路径
        strict: 是否严格匹配参数
        map_location: 强制设备映射
        
    Returns:
        tuple: (缺失参数列表, 意外参数列表)
    """
    path = Path(checkpoint_path).expanduser().resolve()
    LOGGER.debug(f"开始加载检查点: {path}")
    
    if not path.exists():
        LOGGER.critical(f"检查点文件不存在: {path}")
        raise FileNotFoundError(f"文件不存在: {path}")
        
    try:
        # 自动选择设备
        device = map_location or CONFIG.DEVICE
        checkpoint = torch.load(path, map_location=device)
        
        # 处理分布式训练参数
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            
        # 参数名称对齐
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                k = k[7:]  # 去除DDP前缀
            state_dict[k] = v
            
        # 加载参数
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        # 日志记录
        if missing:
            LOGGER.warning(f"缺失参数: {len(missing)}个\n示例: {missing[:3]}")
        if unexpected:
            LOGGER.warning(f"意外参数: {len(unexpected)}个\n示例: {unexpected[:3]}")
            
        LOGGER.info(f"成功加载检查点: {path}")
        return missing, unexpected
        
    except Exception as e:
        LOGGER.critical(f"加载失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"加载检查点失败: {path}") from e

def save_checkpoint(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    overwrite: bool = False,
    metadata: Optional[dict] = None
) -> Path:
    """增强版模型保存（支持元数据存储）
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
        overwrite: 是否覆盖
        metadata: 额外元数据
        
    Returns:
        实际保存路径
    """
    path = Path(save_path).expanduser().resolve()
    LOGGER.debug(f"开始保存检查点到: {path}")
    
    if path.exists() and not overwrite:
        LOGGER.error(f"文件已存在: {path}")
        raise FileExistsError(f"文件已存在，使用overwrite=True强制覆盖")
        
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
            
        # 添加元数据
        save_data = {
            "state_dict": state_dict,
            "config": vars(CONFIG),  # 保存当前配置
            "metadata": metadata or {}
        }
        
        torch.save(save_data, path)
        LOGGER.info(f"模型保存成功: {path}")
        return path
        
    except Exception as e:
        LOGGER.critical(f"保存失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"保存失败: {path}") from e

def set_seed(seed: Optional[int] = None) -> None:
    """增强版随机种子设置（支持多GPU/第三方库）"""
    final_seed = seed or CONFIG.SEED
    deterministic = getattr(CONFIG, "DETERMINISTIC", False)
    
    try:
        import numpy as np
        import random
        
        # 设置基础种子
        torch.manual_seed(final_seed)
        np.random.seed(final_seed)
        random.seed(final_seed)
        
        # CUDA相关设置
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(final_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                LOGGER.warning("启用确定性模式，可能影响性能")
                
        LOGGER.info(f"全局种子设置为: {final_seed} (deterministic={deterministic})")
        
    except Exception as e:
        LOGGER.error(f"设置种子失败: {str(e)}", exc_info=True)
        raise

def get_hardware_utilization() -> Dict[str, str]:
    """增强版硬件监控（支持多GPU）"""
    stats = {}
    
    # CPU信息
    try:
        import psutil
        stats.update({
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "ram_usage": f"{psutil.virtual_memory().percent}%"
        })
    except ImportError:
        LOGGER.warning("需要安装psutil来获取完整CPU信息")
    
    # GPU信息
    if torch.cuda.is_available():
        try:
            device = torch.device(CONFIG.DEVICE)
            torch.cuda.synchronize(device)
            
            # 显存信息
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            stats["gpu_memory"] = f"{allocated:.2f}/{reserved:.2f}/{total:.2f} GB (使用/保留/总量)"
            
            # 计算利用率
            utilization = torch.cuda.utilization(device) if hasattr(torch.cuda, "utilization") else "N/A"
            stats["gpu_util"] = f"{utilization}%"
            
        except Exception as e:
            stats["gpu_error"] = str(e)
            LOGGER.warning(f"获取GPU信息失败: {str(e)}")
    else:
        stats["gpu"] = "不可用"
        
    return stats

def enable_tf32_precision() -> None:
    """智能启用TF32精度"""
    if not torch.cuda.is_available():
        LOGGER.warning("TF32需要NVIDIA GPU支持")
        return
        
    try:
        # 检查架构支持
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            LOGGER.warning("TF32需要Ampere+架构 (RTX 30系列及以上)")
            return
            
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOGGER.info("已启用TF32精度模式")
    except Exception as e:
        LOGGER.error(f"启用TF32失败: {str(e)}", exc_info=True)
        raise

def model_size_in_mb(model: torch.nn.Module) -> float:
    """精确模型大小计算（包含量化支持）"""
    param_size = 0
    for param in model.parameters():
        elem_size = param.element_size()  # 自动处理数据类型
        param_size += param.numel() * elem_size
        
    buffer_size = 0
    for buffer in model.buffers():
        elem_size = buffer.element_size()
        buffer_size += buffer.numel() * elem_size
        
    total_mb = (param_size + buffer_size) / 1024**2
    return round(total_mb, 2)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """优化版因果掩码生成"""
    try:
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)
        return mask.float().masked_fill(mask, float('-inf'))
    except RuntimeError as e:
        LOGGER.error(f"生成掩码失败: {str(e)}")
        raise