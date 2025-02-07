
# 佛祖保佑，代码永无bug，报错
#                            _ooOoo_  
#                           o8888888o  
#                           88" . "88  
#                           (| -_- |)  
#                            O\ = /O  
#                        ____/`---'\____  
#                      .   ' \\| |// `.  
#                       / \\||| : |||// \  
#                     / _||||| -:- |||||- \  
#                       | | \\\ - /// | |  
#                     | \_| ''\---/'' | |  
#                      \ .-\__ `-` ___/-. /  
#                   ___`. .' /--.--\ `. . __  
#                ."" '< `.___\_<|>_/___.' >'"".  
#               | | : `- \`.;`\ _ /`;.`/ - ` : | |  
#                 \ \ `-. \_ __\ /__ _/ .-` / /  
#         ======`-.____`-.___\_____/___.-`____.-'======  
#                            `=---='  
#  
#         .............................................  
#                  佛祖保佑             永无BUG 
#          佛曰:  
#                  写字楼里写字间，写字间里程序员；  
#                  程序人员写程序，又拿程序换酒钱。  
#                  酒醒只在网上坐，酒醉还来网下眠；  
#                  酒醉酒醒日复日，网上网下年复年。  
#                  但愿老死电脑间，不愿鞠躬老板前；  
#                  奔驰宝马贵者趣，公交自行程序员。  
#                  别人笑我忒疯癫，我笑自己命太贱；  
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from config import CONFIG


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, min_len: int = 5000, device=None):
        """
        位置编码模块
        Args:
            d_model: 嵌入维度
            dropout: dropout概率
            min_len: 最小位置编码长度
            device: 设备类型
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.min_len = min_len

        # 注册缓冲区时会自动同步设备
        self.register_buffer('pe', torch.zeros(1, min_len, d_model))
        self._init_pe(min_len)

    def _init_pe(self, max_len: int):
        """初始化位置编码"""
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(1, max_len, self.d_model)
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)
        self.pe[:, :max_len, :] = pe  # 直接赋值

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len) # 如果需要，实现扩展逻辑
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

    def _extend_pe(self, new_len: int):
        """扩展位置编码长度"""
        # 计算新的位置编码
        position = torch.arange(new_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(10000.0) / self.d_model)
        )
        new_pe = torch.zeros(1, new_len, self.d_model)
        new_pe[..., 0::2] = torch.sin(position * div_term)
        new_pe[..., 1::2] = torch.cos(position * div_term)

        # 将新的位置编码注册为缓冲区
        self.register_buffer('pe', new_pe)


class ExpertLayer(nn.Module):
    def __init__(self, num_experts: int, expert_dim: int, d_model: int,
                 balance_coef: float = 0.01):
        """
        MoE专家层
        Args:
            num_experts: 专家数量
            expert_dim: 专家网络维度
            d_model: 模型维度
            balance_coef: 负载均衡系数
        """
        super().__init__()
        self.num_experts = num_experts
        self.balance_coef = balance_coef

        # 使用ModuleList自动同步设备
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        """前向传播"""
        bs, seq_len, d_model = x.size()
        flat_x = x.view(-1, d_model)

        # 路由计算
        gate_logits = self.gate(flat_x)
        routing_weights = F.softmax(gate_logits, dim=-1)
        expert_choice = torch.argmax(routing_weights, dim=-1)

        # 负载均衡损失
        expert_mask = F.one_hot(expert_choice, num_classes=self.num_experts).float()
        expert_counts = expert_mask.sum(dim=0)
        expert_probs = expert_counts / (bs * seq_len)
        balance_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-10)) * self.balance_coef

        # 并行处理专家网络
        expert_outputs = [self.experts[i](flat_x) for i in range(self.num_experts)]  # 使用列表推导式
        expert_outputs = torch.stack(expert_outputs, dim=1)
        selected_outputs = expert_outputs[torch.arange(flat_x.size(0)), expert_choice]

        projected = self.proj(selected_outputs)
        return self.dropout(projected.view(bs, seq_len, d_model)), balance_loss


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=None, embed_size=None, num_heads=None, hidden_dim=None,
                 enc_layers=None, dec_layers=None, dropout=None, use_moe=None,
                 num_experts=None, expert_dim=None, device=None):
        super().__init__()
        # 参数初始化
        cfg = CONFIG
        self.vocab_size = vocab_size or cfg.VOCAB_SIZE
        self.embed_size = embed_size or cfg.EMBED_SIZE
        self.num_heads = num_heads or cfg.NUM_HEADS
        self.hidden_dim = hidden_dim or cfg.HIDDEN_DIM
        self.enc_layers = enc_layers or cfg.ENC_LAYERS
        self.dec_layers = dec_layers or cfg.DEC_LAYERS
        self.dropout = dropout or cfg.DROPOUT
        self.use_moe = use_moe if use_moe is not None else cfg.USE_MOE
        self.num_experts = num_experts or cfg.NUM_EXPERTS
        self.expert_dim = expert_dim or cfg.EXPERT_DIM
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # === 核心修复：统一设备管理 ===
        # 1. 移除所有显式的.to(device)调用，依赖PyTorch的自动设备管理
        # 2. 使用register_buffer替代手动设备分配
        # 3. 确保所有子模块继承父模块的设备

        # --- 嵌入层 ---
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.pos_encoder = PositionalEncoding(
            d_model=self.embed_size,
            dropout=self.dropout,
        )

        # --- Transformer 编码器/解码器 ---
        # 创建时不需要指定设备，通过父模块统一管理
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.dec_layers)

        # --- MoE 专家层 ---
        self.moe_layers = nn.ModuleDict()  # 改用ModuleDict自动管理设备
        if self.use_moe:
            self.moe_layers["encoder"] = nn.ModuleList([
                ExpertLayer(
                    num_experts=self.num_experts,
                    expert_dim=self.expert_dim,
                    d_model=self.embed_size,
                    balance_coef=0.01
                ) for _ in range(self.enc_layers)
            ])
            self.moe_layers["decoder"] = nn.ModuleList([
                ExpertLayer(
                    num_experts=self.num_experts,
                    expert_dim=self.expert_dim,
                    d_model=self.embed_size,
                    balance_coef=0.01
                ) for _ in range(self.dec_layers)
            ])

        # --- 最后的全连接层 ---
        self.fc = nn.Linear(self.embed_size, self.vocab_size)

        # 初始化权重并统一设备
        self._init_weights()
        self.to(self.device)  # 统一移动整个模型到指定设备

    def _init_weights(self):
        """使用Kaiming初始化优化参数分布"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embed' in name:
                    nn.init.normal_(param, mean=0, std=self.embed_size ** -0.5)
                elif 'linear' in name.lower():
                    # nn.init.kaiming_normal_(param, nonlinearity='gelu') # 不支持 gelu
                     nn.init.xavier_normal_(param)  # 或者使用 xavier_normal_
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        前向传播流程优化：
        1. 自动处理设备转换
        2. 添加维度检查
        3. 优化计算图构建
        """
        # === 设备一致性检查 ===
        # src, tgt = src.to(self.device), tgt.to(self.device) #数据准备阶段处理

        # === 输入合法性检查 ===
        assert src.dim() == 2, f"输入src维度错误，应为(batch, seq_len)，实际得到{src.shape}"
        assert tgt.dim() == 2, f"输入tgt维度错误，应为(batch, seq_len)，实际得到{tgt.shape}"

        # === 调试信息 ===
        print(f"src max: {src.max()}, src min: {src.min()}, vocab_size: {self.vocab_size}")

        # === 裁剪索引（临时解决方案）===
        src = torch.clamp(src, 0, self.vocab_size - 1)
        tgt = torch.clamp(tgt, 0, self.vocab_size - 1) #裁剪tgt

        # === 嵌入层 ===
        src_embed = self.embed(src) * math.sqrt(self.embed_size)  # 缩放嵌入
        src_embed = self.pos_encoder(src_embed)
        tgt_embed = self.embed(tgt) * math.sqrt(self.embed_size)
        tgt_embed = self.pos_encoder(tgt_embed)

        # 编码器处理
        memory = src_embed
        balance_losses = []
        for i in range(self.enc_layers):
            # 应用梯度检查点
            memory = checkpoint.checkpoint(self.encoder.layers[i], memory, src_mask,
                                           src_key_padding_mask)  # 注意这里的参数顺序必须与 forward 函数的参数顺序一致
            if self.use_moe and "encoder" in self.moe_layers:
                memory, bl = self.moe_layers["encoder"][i](memory)
                balance_losses.append(bl)

        # 解码器处理
        output = tgt_embed
        for i in range(self.dec_layers):
            # 应用梯度检查点
            output = checkpoint.checkpoint(self.decoder.layers[i], output, memory, tgt_mask, memory_mask,
                                           tgt_key_padding_mask,
                                           memory_key_padding_mask)  # 注意这里的参数顺序必须与 forward 函数的参数顺序一致
            if self.use_moe and "decoder" in self.moe_layers:
                output, bl = self.moe_layers["decoder"][i](output)
                balance_losses.append(bl)

        # === 输出处理 ===
        output = self.fc(output)
        total_balance_loss = torch.sum(torch.stack(balance_losses)) if balance_losses else 0.0

        return output, total_balance_loss
