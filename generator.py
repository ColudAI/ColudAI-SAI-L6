import torch
import math
from typing import Optional
from tokenizers import Tokenizer
from torch import Tensor
from config import CONFIG, LOGGER  # 使用新的配置系统

def generate_response(
    model: TransformerModel,
    tokenizer: Tokenizer,
    input_text: str,
    generation_config: Optional[dict] = None
) -> str:
    """
    生成对话响应的完整实现，适配最新配置系统
    
    参数:
        model: 加载的Transformer模型
        tokenizer: 分词器实例
        input_text: 输入的对话文本
        generation_config: 可选的生成参数覆盖配置
        
    返回:
        生成的响应文本
    """
    # 合并默认配置和自定义配置
    config = {
        "max_length": CONFIG.GENERATION_MAX_LENGTH,
        "temperature": CONFIG.GENERATION_TEMPERATURE,
        "top_k": CONFIG.GENERATION_TOP_K,
        "top_p": CONFIG.GENERATION_TOP_P,
        "device": CONFIG.DEVICE,
        "eos_token": CONFIG.EOS_TOKEN,
        "bos_token": CONFIG.BOS_TOKEN,
        "sep_token": CONFIG.SEP_TOKEN,
        "pad_token": CONFIG.PAD_TOKEN
    }
    if generation_config:
        config.update(generation_config)
    
    LOGGER.debug(f"生成配置: {config}")

    # 准备输入序列
    formatted_input = f"{config['bos_token']}{input_text}{config['sep_token']}"
    input_ids = tokenizer.encode(formatted_input).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=config["device"])

    # 编码阶段
    with torch.no_grad():
        src_embed = model.embed(input_tensor) * math.sqrt(model.embed_size)
        src_embed = model.pos_encoder(src_embed)
        memory = model.encoder(src_embed)

    # 解码初始化
    eos_token_id = tokenizer.token_to_id(config["eos_token"])
    current_ids = torch.tensor(
        [[tokenizer.token_to_id(config["bos_token"])]], 
        dtype=torch.long, 
        device=config["device"]
    )

    # 生成循环
    for _ in range(config["max_length"]):
        with torch.no_grad():
            # 准备解码器输入
            tgt_embed = model.embed(current_ids) * math.sqrt(model.embed_size)
            tgt_embed = model.pos_encoder(tgt_embed)  # 使用统一的位置编码
            
            # 生成注意力掩码
            tgt_mask = generate_square_subsequent_mask(current_ids.size(1), config["device"])
            
            # 解码步骤
            output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = model.fc(output[:, -1, :]) / config["temperature"]

            # 概率处理
            probs = process_probs(logits, config)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # 终止条件检查
            if next_token.item() == eos_token_id:
                break

    # 解码结果并清理特殊标记
    decoded = tokenizer.decode(current_ids[0].tolist(), skip_special_tokens=True)
    return decoded.strip()

def process_probs(logits: Tensor, config: dict) -> Tensor:
    """概率处理管道，包含Top-K和Top-P采样"""
    probs = torch.softmax(logits, dim=-1)
    
    # Top-K过滤
    if config["top_k"] > 0:
        top_k = min(config["top_k"], probs.size(-1))
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
    
    # Top-P过滤
    if 0 < config["top_p"] < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除累计概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > config["top_p"]
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 应用掩码并重新归一化
        probs = probs.masked_fill(sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove), 0.0)
    
    # 确保概率有效
    if torch.any(torch.isnan(probs)):
        LOGGER.warning("检测到NaN概率，启用安全模式")
        probs = torch.ones_like(probs) / probs.size(-1)
    
    return probs / probs.sum(dim=-1, keepdim=True)

def generate_square_subsequent_mask(sz: int, device: torch.device) -> Tensor:
    """生成因果掩码的优化实现"""
    return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)