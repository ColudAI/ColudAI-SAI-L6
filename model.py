import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TrainingConfig

class PositionalEncoding(nn.Module):
    """增强型位置编码，支持动态扩展"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.register_buffer('pe', None)
        self.max_len = 0

    def _update_pe(self, max_len: int):
        """动态更新位置编码缓存"""
        if self.pe is None or max_len > self.max_len:
            self.max_len = max(max_len, 5000)  # 最小缓存5000
            pe = torch.zeros(self.max_len, self.d_model)
            position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2).float() 
                * (-math.log(10000.0) / self.d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._update_pe(x.size(1))
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ExpertLayer(nn.Module):
    """带负载均衡的MoE专家层"""
    def __init__(self, d_model: int, num_experts: int, expert_dim: int):
        super().__init__()
        self.num_experts = num_experts
        cfg = TrainingConfig()
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim * 2),
                nn.GLU(dim=-1),
                nn.LayerNorm(expert_dim),
                nn.Dropout(cfg.DROPOUT)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model, num_experts * 2),
            nn.GLU(dim=-1),
            nn.Linear(num_experts, num_experts, bias=False)
        )
        
        # 投影层
        self.proj = nn.Linear(expert_dim, d_model)
        self.dropout = nn.Dropout(cfg.DROPOUT)
        self.balance_coef = cfg.BALANCE_COEF

    def forward(self, x: torch.Tensor):
        bs, seq_len, d_model = x.size()
        flat_x = x.view(-1, d_model)
        
        # 门控计算
        gate_logits = self.gate(flat_x)
        routing_weights = F.softmax(gate_logits, dim=-1)
        expert_choice = torch.argmax(routing_weights, dim=-1)
        
        # 负载均衡损失
        expert_counts = torch.bincount(expert_choice, minlength=self.num_experts).float()
        expert_probs = expert_counts / (bs * seq_len)
        balance_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10)) * self.balance_coef
        
        # 并行处理专家网络
        expert_outputs = torch.stack([expert(flat_x) for expert in self.experts], dim=1)
        
        # 选择专家输出
        mask = F.one_hot(expert_choice, self.num_experts).bool()
        selected_outputs = expert_outputs[mask.unsqueeze(-1).expand_as(expert_outputs)]
        selected_outputs = selected_outputs.view(-1, self.experts[0].out_features)
        
        projected = self.proj(selected_outputs)
        return self.dropout(projected.view(bs, seq_len, d_model)), balance_loss

class TransformerModel(nn.Module):
    """支持MoE的Transformer模型"""
    def __init__(self):
        super().__init__()
        cfg = TrainingConfig()
        
        # 嵌入层
        self.embed = nn.Embedding(cfg.VOCAB_SIZE, cfg.EMBED_SIZE)
        self.pos_encoder = PositionalEncoding(cfg.EMBED_SIZE, cfg.DROPOUT)
        
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.EMBED_SIZE,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_DIM,
            dropout=cfg.DROPOUT,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.ENC_LAYERS)
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.EMBED_SIZE,
            nhead=cfg.NUM_HEADS,
            dim_feedforward=cfg.HIDDEN_DIM,
            dropout=cfg.DROPOUT,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.DEC_LAYERS)
        
        # MoE模块
        self.moe_layers = nn.ModuleDict()
        if cfg.USE_MOE:
            self.moe_layers["encoder"] = nn.ModuleList([
                ExpertLayer(cfg.EMBED_SIZE, cfg.NUM_EXPERTS, cfg.EXPERT_DIM)
                for _ in range(cfg.ENC_LAYERS)
            ])
            self.moe_layers["decoder"] = nn.ModuleList([
                ExpertLayer(cfg.EMBED_SIZE, cfg.NUM_EXPERTS, cfg.EXPERT_DIM)
                for _ in range(cfg.DEC_LAYERS)
            ])
        
        # 输出层
        self.fc = nn.Linear(cfg.EMBED_SIZE, cfg.VOCAB_SIZE)
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.embed.embedding_dim**-0.5)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # 生成注意力掩码
        src_mask = self._generate_square_mask(src.size(1), src.device)
        tgt_mask = self._generate_square_mask(tgt.size(1), tgt.device)
        
        # 编码处理
        src_emb = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, mask=src_mask)
        
        # 解码处理
        tgt_emb = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # MoE处理
        balance_loss = 0
        if self.moe_layers:
            # 编码器MoE
            for moe in self.moe_layers["encoder"]:
                memory, bl = moe(memory)
                balance_loss += bl
            
            # 解码器MoE
            for moe in self.moe_layers["decoder"]:
                output, bl = moe(output)
                balance_loss += bl
        
        return self.fc(output), balance_loss

    @staticmethod
    def _generate_square_mask(sz: int, device) -> torch.Tensor:
        """生成因果掩码"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)