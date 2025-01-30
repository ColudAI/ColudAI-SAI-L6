# model_loader.py
import torch
from typing import Tuple
from tokenizer import BpeTokenizer
from chat_model import TransformerModel
from config import Config
import logging

def resize_embeddings(model: TransformerModel, new_vocab_size: int):
    old_vocab_size = model.embed.num_embeddings
    if new_vocab_size == old_vocab_size:
        return

    logging.info(f"Resizing embeddings from {old_vocab_size} to {new_vocab_size}")
    
    # Resize embedding layer
    new_embed = torch.nn.Embedding(new_vocab_size, model.embed.embedding_dim)
    new_embed.weight.data[:old_vocab_size, :] = model.embed.weight.data
    torch.nn.init.xavier_uniform_(new_embed.weight.data[old_vocab_size:])
    model.embed = new_embed
    
    # Resize output layer
    new_fc = torch.nn.Linear(model.fc.in_features, new_vocab_size)
    new_fc.weight.data[:, :old_vocab_size] = model.fc.weight.data[:, :old_vocab_size]
    new_fc.bias.data[:old_vocab_size] = model.fc.bias.data
    torch.nn.init.xavier_uniform_(new_fc.weight.data[:, old_vocab_size:])
    new_fc.bias.data[old_vocab_size:].fill_(0)
    model.fc = new_fc

def load_model(config: Config) -> Tuple[TransformerModel, BpeTokenizer]:
    tokenizer = BpeTokenizer(str(config.tokenizer_path))
    
    # 加载模型状态
    state_dict = torch.load(config.model_path, map_location=config.device)

    vocab_size = state_dict['embed.weight'].shape[0]
    model = TransformerModel(
        vocab_size=5722,      # 初始词汇量
        embed_size=config.embed_size,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        enc_layers=config.enc_layers,
        dec_layers=config.dec_layers
    )

    model.load_state_dict(state_dict, strict=False)
    model.to(config.device)

    new_vocab_size = len(tokenizer.vocab)
    if new_vocab_size != vocab_size:
        resize_embeddings(model, new_vocab_size)
    
    model.eval()
    return model, tokenizer

def optimize_model_for_inference(model: TransformerModel, config: Config) -> TransformerModel:
    """使用TorchScript优化模型"""
    example_input = torch.zeros(1, config.max_length, dtype=torch.long).to(config.device)
    traced_model = torch.jit.trace(model, (example_input, example_input))
    traced_model = torch.jit.optimize_for_inference(traced_model)
    logging.info("模型已通过TorchScript优化")
    return traced_model