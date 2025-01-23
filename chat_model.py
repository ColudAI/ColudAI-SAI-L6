import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerDecoderModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embed(src) * torch.sqrt(torch.tensor(src.size(-1)).float())
        tgt = self.embed(tgt) * torch.sqrt(torch.tensor(tgt.size(-1)).float())
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out = self.transformer_decoder(tgt, src)
        out = self.fc(out)
        return out

class ChatDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']

        question_encoding = self.tokenizer.encode(question)
        answer_encoding = self.tokenizer.encode(answer)

        question_ids = question_encoding.ids
        answer_ids = answer_encoding.ids

        system_token_id = self.tokenizer.token_to_id("<|system|>")

        input_ids = question_ids + [system_token_id] + answer_ids[:-1]
        label_ids = answer_ids[1:]

        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]

        input_ids = input_ids + [0] * (self.max_len - len(input_ids))
        label_ids = label_ids + [-100] * (self.max_len - len(label_ids))

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def train_model(data_path, tokenizer_path, model_path, vocab_size, min_freq, epochs=10, batch_size=2, grad_accum_steps=16):
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("Loaded existing tokenizer.")
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq)
        tokenizer.train([data_path], trainer)
        tokenizer.add_special_tokens(["<|system|>"])  # 添加特殊标记
        tokenizer.save(tokenizer_path)
        print("Trained and saved tokenizer.")

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocab size: {actual_vocab_size}")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']

    dataset = ChatDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerDecoderModel(actual_vocab_size, embed_size=128, num_heads=2, hidden_dim=256, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        print(f'\nEpoch {epoch+1}/{epochs}\n')
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, input_ids)
            loss = criterion(outputs.view(-1, actual_vocab_size), labels.view(-1))
            loss = loss / grad_accum_steps
            loss.backward()

            if (i + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            progress_bar.set_postfix(loss=loss.item())

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    torch.save(model.state_dict(), model_path)

def load_model(model_path, tokenizer_path):
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file not found at {tokenizer_path}. Training a new tokenizer.")
        train_model(data_path, tokenizer_path, model_path, vocab_size, min_freq)
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Actual vocab size: {actual_vocab_size}")

    embed_size = 128
    num_heads = 2
    hidden_dim = 256
    num_layers = 2

    model = TransformerDecoderModel(actual_vocab_size, embed_size, num_heads, hidden_dim, num_layers)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(checkpoint, strict=False)

    return model, tokenizer

if __name__ == "__main__":
    data_path = 'train_data.json'
    tokenizer_path = 'tokenizer.json'
    model_path = 'chat_model.pth'
    vocab_size = 10000
    min_freq = 2
    train_model(data_path, tokenizer_path, model_path, vocab_size, min_freq)

    model, tokenizer = load_model(model_path, tokenizer_path)