#        ┏┓　　　┏┓+ +
#　　　┏┛┻━━━┛┻┓ + +
#　　　┃　　　　　　　┃ 　
#　　　┃　　　━　　　┃ ++ + + +
#　　 ████━████ ┃+
#　　　┃　　　　　　　┃ +
#　　　┃　　　┻　　　┃
#　　　┃　　　　　　　┃ + +
#　　　┗━┓　　　┏━┛
#　　　　　┃　　　┃　　　　　　　　　　　
#　　　　　┃　　　┃ + + + +
#　　　　　┃　　　┃　　　　Codes are far away from bugs with the animal protecting　　　
#　　　　　┃　　　┃ + 　　　　神兽保佑,代码永无bug及报错
#　　　　　┃　　　┃
#　　　　　┃　　　┃　　+　　　　　　　　　
#　　　　　┃　 　　┗━━━┓ + +
#　　　　　┃ 　　　　　　　┣┓
#　　　　　┃ 　　　　　　　┏┛
#　　　　　┗┓┓┏━┳┓┏┛ + + + +
#　　　　　　┃┫┫　┃┫┫
#　　　　　　┗┻┛　┗┻┛+ + + +
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 统一配置管理
from config import CONFIG, LOGGER
from model import TransformerModel
from data import DataLoaderPreparer

def data_preparation(batch: dict) -> dict:
    device = CONFIG.DEVICE
    inputs = batch["input_ids"].to(device, non_blocking=True)

    # 显式对齐所有输出设备
    return {
        "inputs": inputs.contiguous(),
        "dec_input": inputs[:, :-1].contiguous(),
        "dec_output": inputs[:, 1:].contiguous()
    }

def forward_pass(model: TransformerModel, batch_data: dict, criterion: torch.nn.Module) -> torch.Tensor:
    """优化后的前向传播，使用配置缓存"""
    # 缓存常用配置参数
    amp_enabled = CONFIG.USE_AMP
    amp_dtype = torch.bfloat16 if CONFIG.USE_BF16 else torch.float16
    balance_coef = CONFIG.BALANCE_COEF
    grad_accum = CONFIG.GRAD_ACCUM
    device = CONFIG.DEVICE  # 获取设备

    # 确保数据在正确的设备上
    inputs = batch_data["inputs"].to(device)
    dec_input = batch_data["dec_input"].to(device)
    dec_output = batch_data["dec_output"].to(device)

    with autocast(enabled=amp_enabled, dtype=amp_dtype):
        outputs, balance_loss = model(inputs, dec_input)
        logits = outputs.view(-1, outputs.size(-1))
        targets = dec_output.view(-1)

        # 检查并裁剪目标值
        print(f"Targets max: {targets.max()}, min: {targets.min()}, vocab_size: {model.vocab_size}")
        targets = torch.clamp(targets, 0, model.vocab_size - 1)  # 裁剪目标值

        loss_mask = targets != -100
        loss = criterion(logits[loss_mask], targets[loss_mask])
        return (loss + balance_loss * balance_coef) / grad_accum

def backward_and_update(
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        total_loss: torch.Tensor,
        step: int,
        total_steps: int
):
    """优化后的反向传播与参数更新"""
    grad_accum = CONFIG.GRAD_ACCUM
    clip_grad_norm = CONFIG.CLIP_GRAD_NORM
    debug_mode = CONFIG.DEBUG

    scaler.scale(total_loss).backward()

    if (step + 1) % grad_accum == 0 or (step + 1) == total_steps:
        scaler.unscale_(optimizer)

        # 梯度裁剪优化
        for group in optimizer.param_groups:
            if group.get("no_clip", False):
                continue
            torch.nn.utils.clip_grad_norm_(
                group["params"],
                group["lr"],  # 添加 lr 参数
                error_if_nonfinite=debug_mode
            )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


def train_epoch(
        model: TransformerModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: torch.nn.Module,
        scaler: GradScaler,
        epoch: int
) -> float:
    """优化后的训练周期，改进进度显示和内存管理"""
    model.train()
    total_loss = 0.0
    accumulated_steps = 0

    # 缓存配置参数
    grad_accum = CONFIG.GRAD_ACCUM
    total_epochs = CONFIG.EPOCHS
    warmup_steps = CONFIG.WARMUP_STEPS

    with tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"🚀 训练周期 [{epoch + 1}/{total_epochs}]",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ) as pbar:
        for step, batch in pbar:
            try:
                batch_data = data_preparation(batch)
                loss = forward_pass(model, batch_data, criterion)

                backward_and_update(
                    optimizer=optimizer,
                    scaler=scaler,
                    total_loss=loss,
                    step=step,
                    total_steps=len(train_loader)
                )

                # 学习率调度优化
                if (step + 1) % grad_accum == 0:
                    scheduler.step()

                total_loss += loss.item() * grad_accum
                accumulated_steps += 1

                # 实时监控指标更新
                current_lr = optimizer.param_groups[0]["lr"]
                mem_usage = torch.cuda.memory_allocated() // 1024 ** 2 if torch.cuda.is_available() else 0
                pbar.set_postfix({
                    "loss": f"{total_loss / accumulated_steps:.3f}",
                    "lr": f"{current_lr:.2e}",
                    "gpu_mem": f"{mem_usage}MB"
                })

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    LOGGER.error("显存不足，尝试减小批次大小或启用梯度检查点")
                    torch.cuda.empty_cache()
                    continue
                raise

    avg_loss = total_loss / accumulated_steps if accumulated_steps else 0.0
    LOGGER.info(f"训练周期 {epoch + 1} 完成 | 平均损失: {avg_loss:.4f}")
    return avg_loss


def setup_optimizer(model: TransformerModel) -> torch.optim.Optimizer:
    """优化器配置改进"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.BASE_LR,
        eps=1e-8,
        weight_decay=CONFIG.WEIGHT_DECAY
    )


def setup_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    """学习率调度器优化"""
    warmup_steps = CONFIG.WARMUP_STEPS
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
    )


def main():
    """主函数优化，改进初始化流程"""
    # 设备选择
    device = CONFIG.DEVICE
    CONFIG.DEVICE = device

    # 模型初始化
    model = TransformerModel(
        vocab_size=CONFIG.VOCAB_SIZE,
        embed_size=CONFIG.EMBED_SIZE,
        num_heads=CONFIG.NUM_HEADS,
        hidden_dim=CONFIG.HIDDEN_DIM,
        enc_layers=CONFIG.ENC_LAYERS,
        dec_layers=CONFIG.DEC_LAYERS,
        dropout=CONFIG.DROPOUT,
        use_moe=CONFIG.USE_MOE,
        num_experts=CONFIG.NUM_EXPERTS,
        expert_dim=CONFIG.EXPERT_DIM,
        device=device
    )

    print(f"模型最终设备: {next(model.parameters()).device}")
    print(f"Embedding层权重设备: {model.embed.weight.device}")

    # 数据加载器初始化
    dl_preparer = DataLoaderPreparer()
    train_loader, val_loader = dl_preparer.prepare()

    # 检查点恢复
    if CONFIG.MODEL_SAVE_PATH.exists():
        model.load_state_dict(torch.load(CONFIG.MODEL_SAVE_PATH, map_location=device))
        LOGGER.info(f"成功加载检查点: {CONFIG.MODEL_SAVE_PATH}")

    # 训练组件初始化
    optimizer = setup_optimizer(model)
    scheduler = setup_scheduler(optimizer)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    scaler = GradScaler(enabled=CONFIG.USE_AMP)

    # 确保保存目录存在
    CONFIG.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 训练循环
    for epoch in range(CONFIG.EPOCHS):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
        checkpoint_path = CONFIG.MODEL_SAVE_PATH.parent / f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        LOGGER.info(f"检查点已保存至: {checkpoint_path}")


if __name__ == "__main__":
    main()