import os
import torch
import torch.distributed as dist
from config import CONFIG, LOGGER

def setup_distributed() -> tuple:
    """
    初始化分布式训练环境
    
    返回:
        (is_ddp, rank, world_size): 元组包含:
            - is_ddp: 是否启用分布式训练
            - rank: 当前进程排名
            - world_size: 总进程数
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # 设置CUDA设备
        torch.cuda.set_device(rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 配置日志只在主进程输出
        if rank != 0:
            LOGGER.setLevel(logging.WARNING)
            
        return True, rank, world_size
    
    # 非分布式模式
    return False, 0, 1

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def wrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    将模型包装为分布式数据并行模式
    
    参数:
        model: 原始模型
        
    返回:
        包装后的DDP模型
    """
    if CONFIG.DEVICE == 'cuda' and dist.is_initialized():
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[CONFIG.DEVICE],
            output_device=CONFIG.DEVICE,
            find_unused_parameters=False
        )
    return model

def get_distributed_sampler(dataset: torch.utils.data.Dataset) -> torch.utils.data.Sampler:
    """
    创建分布式采样器
    
    参数:
        dataset: 目标数据集
        
    返回:
        分布式采样器实例
    """
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=int(os.environ.get('WORLD_SIZE', 1)),
        rank=int(os.environ.get('RANK', 0)),
        shuffle=True,
        drop_last=True
    )