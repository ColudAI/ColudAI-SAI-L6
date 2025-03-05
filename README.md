
# 基于Transformer的MoE模型实现
<div align="center">
 <img src="https://raw.githubusercontent.com/ColudAI/ColudAI-SAI-L6/refs/heads/main/%E5%AE%B9%E5%99%A8%201%401x%20(7).png" width="60%" alt="ColludAI " />
</div>
<hr>
本仓库包含一个基于Transformer的混合专家模型（Mixture of Experts, MoE）的实现，使用PyTorch和tokenizers库。它包括位置编码、专家层以及完整的Transformer架构（编码器和解码器）等模块。

## 概览

本项目结构如下：

*   **`config.py`**: 配置文件，包含模型和训练过程的超参数和设置。*请参考代码中的注释了解更多信息。*
*   **`model.py`**: 定义Transformer模型架构，包括位置编码、MoE层、以及编码器/解码器结构。
    ```python
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size=None, embed_size=None, ...):
            super().__init__()
            # 参数初始化
            cfg = CONFIG
            self.vocab_size = vocab_size or cfg.VOCAB_SIZE
            self.embed = nn.Embedding(self.vocab_size, self.embed_size)
            # ... 其他层定义
        def forward(self, src, tgt, ...):
            # 前向传播逻辑
            output = self.fc(output)
            return output, total_balance_loss
    ```
*   **`tokenizer.py`**: 实现一个用于文本处理的自定义BPE分词器。
*   **`train.py`**: 包含训练循环、数据加载、优化和模型保存逻辑。
*   **`data.py`**: 包含用于训练模型的数据加载和预处理函数。

## `train.py`: 训练脚本

此脚本处理模型训练过程。 以下是主要组件和优化的细分：

### 1. 配置和设置：

*   导入必要的库（PyTorch、tqdm、pathlib、自定义模块）。
*   从 `config.py` 加载配置设置。
*   初始化 Transformer 模型、数据加载器、优化器、学习率调度器、损失函数和 GradScaler。
*   处理检查点加载以恢复训练。

### 2. 核心功能：

*   **`data_preparation(batch: dict) -> dict`**:
    *   通过将输入张量移动到指定的设备（GPU 或 CPU）并确保正确的张量形状来准备模型的数据。
    *   返回一个包含输入 ID、解码器输入和解码器输出的字典。

*   **`forward_pass(model: TransformerModel, batch_data: dict, criterion: torch.nn.Module) -> torch.Tensor`**:
    *   执行通过模型的前向传递。
    *   应用自动广播（AMP/BF16）以进行混合精度训练。
    *   使用 CrossEntropyLoss 计算损失。
    *   合并 MoE 层的平衡损失。
    ```python
    def forward_pass(model: TransformerModel, batch_data: dict, criterion: torch.nn.Module) -> torch.Tensor:
        with autocast(enabled=CONFIG.USE_AMP, dtype=torch.bfloat16 if CONFIG.USE_BF16 else torch.float16):
            outputs, balance_loss = model(inputs, dec_input)
            loss = criterion(logits[loss_mask], targets[loss_mask])
            return (loss + balance_loss * CONFIG.BALANCE_COEF) / CONFIG.GRAD_ACCUM
    ```
*   **`backward_and_update(...)`**:
    *   执行反向传播并更新模型参数。
    *   应用梯度缩放（使用 GradScaler）进行混合精度训练。
    *   应用梯度裁剪以防止梯度爆炸。
    *   更新优化器和学习率调度器。
*   **`train_epoch(...)`**:
    *   实现一个训练周期。
    *   迭代数据加载器，执行前向传递、后向传递和参数更新。
    *   跟踪训练损失、学习率和 GPU 内存使用情况。
    *   使用 `tqdm` 进行进度条可视化。
    *   处理 CUDA 内存不足错误。
*   **`setup_optimizer(model: TransformerModel) -> torch.optim.Optimizer`**:
    *   使用指定的学习率、权重衰减和其他参数配置 AdamW 优化器。
*   **`setup_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR`**:
    *   配置学习率调度器 (LambdaLR) 以预热学习率。

### 3. 主要训练循环：

*   迭代指定的周期数。
*   为每个周期调用 `train_epoch`。
*   在每个周期后保存模型检查点。

### `tokenizer.py`: BPE 分词器

此模块使用 `tokenizers` 库实现自定义字节对编码 (BPE) 分词器。 它包括用于训练、保存、加载、编码和解码文本的函数。 *请参阅内联注释了解更多信息。*
```python
class BpeTokenizer:
    def __init__(self, path: str = None, special_tokens: List[str] = None):
        # 初始化tokenizer
        pass

    def encode(self, text: str, max_length: int = None, truncation: bool = False) -> List[int]:
        # 编码文本
        encoded: Encoding = self._tokenizer.encode(text)
        return encoded.ids
```

## 使用方法

1.  **安装依赖:**

    ```bash
    pip install torch tqdm tokenizers
    ```

2.  **配置 `config.py`:**

    *   调整超参数（学习率、批大小等）。
    *   设置数据和模型保存的文件路径。
    *   启用/禁用MoE层、AMP和其他功能。

3.  **准备模型参数集:**

    *   将模型的超参数以及其他配置信息，以JSON格式的文件存储到 `data` 文件夹中。文件名可以自定义，但需要在 `config.py` 中正确配置路径。例如，创建一个名为 `model_params.json` 的文件：
        ```json
        {
          "vocab_size": 10000,
          "embed_size": 512,
          "num_heads": 8,
          "hidden_dim": 2048,
          "enc_layers": 6,
          "dec_layers": 6,
          "dropout": 0.1
        }
        ```

4.  **训练模型:**

    ```bash
    python train.py
    ```

## 关键技术和注意事项

*   **设备管理：** 代码已更新，以确保张量和模型参数的一致设备放置。
*   **混合精度训练：** 使用 `torch.cuda.amp` 可以显着加快训练速度，尤其是在具有张量核心的 GPU 上。
*   **梯度累积：** 一种有用的技术，当您受到 GPU 内存的限制并且需要使用更大的有效批量大小时。
*   **MoE 实现：** 混合专家层为模型增加了容量，使其能够学习更复杂的模式。 试验专家数量和专家维度。 监控平衡损失，以确保有效地利用专家。
*   **梯度检查点：** 有助于减少内存占用空间，尤其是对于深度模型，但可能会稍微增加训练时间。
*   **正则化：** 使用 dropout 和权重衰减有助于防止过度拟合。
*   **重要：** 按需配置config.py文件，否则可能无法正常运行！！！

## 未来工作

*   实施验证和测试循环。
*   增加对不同数据集的支持。
*   探索其他MoE路由策略。
*   与分布式训练框架集成。
*   实现思考链等
