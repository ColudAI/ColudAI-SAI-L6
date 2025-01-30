import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from config import TrainingConfig, logger
from model import TransformerModel
from data import prepare_dataloaders

def data_preparation(batch, cfg): 
    """
    准备输入数据，包括移动到指定设备和提取必要的输入输出数据
    """
    inputs = batch["input_ids"].to(cfg.DEVICE, non_blocking=True)
    labels = batch["labels"].to(cfg.DEVICE, non_blocking=True)
    dec_input = inputs[:, :-1]
    dec_output = labels[:, 1:]
    return inputs, dec_input, dec_output

def forward_pass(model, inputs, dec_input, dec_output, criterion, cfg): 
    """
    执行模型的前向传播，包括混合精度计算
    """
    with autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP): 
        outputs, balance_loss = model(inputs, dec_input)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), dec_output.reshape(-1))
        total_loss = (loss + balance_loss * cfg.BALANCE_COEF) / cfg.GRAD_ACCUM
    return total_loss

def backward_and_update(optimizer, scaler, model, total_loss, cfg, step, train_loader): 
    """
    执行反向传播、梯度裁剪和参数更新
    """
    scaler.scale(total_loss).backward()
    if step % cfg.GRAD_ACCUM == 0 or step == len(train_loader): 
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

def train_epoch(
    model: TransformerModel, 
    train_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR, 
    criterion: torch.nn.Module, 
    scaler: GradScaler, 
    epoch: int
) -> float: 
    """
    执行单个训练周期

    参数: 
        model: 待训练模型
        train_loader: 训练数据加载器
        optimizer: 优化器实例
        scheduler: 学习率调度器
        criterion: 损失函数
        scaler: 混合精度梯度缩放器
        epoch: 当前周期数

    返回: 
        avg_loss: 平均训练损失
    """
    cfg = TrainingConfig()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    pbar = tqdm(train_loader, 
                desc=f"🚀 训练周期 {epoch + 1}/{cfg.EPOCHS}", 
                dynamic_ncols=True)

    for step, batch in enumerate(pbar, 1): 
        # 数据准备
        inputs, dec_input, dec_output = data_preparation(batch, cfg)

        # 混合精度前向传播
        loss = forward_pass(model, inputs, dec_input, dec_output, criterion, cfg)

        # 反向传播与梯度裁剪
        backward_and_update(optimizer, scaler, model, loss, cfg, step, train_loader)

        # 更新学习率调度器
        scheduler.step()

        # 累积损失
        total_loss += loss.item() * cfg.GRAD_ACCUM

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss