import torch
import json
from tokenizers import ByteLevelBPETokenizer
from chat_model import TransformerDecoderModel


class BpeTokenizer:
    def __init__(self, tokenizer_path):
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)

        vocab = tokenizer_config['model']['vocab']
        merges = tokenizer_config['model']['merges']
        merges = [tuple(merge.split(' ')) for merge in merges if isinstance(merge, str) and len(merge.split(' ')) == 2]

        self._tokenizer = ByteLevelBPETokenizer(
            vocab=vocab,
            merges=merges
        )

        special_tokens = ['<unk>', '<s>', '</s>']
        self._tokenizer.add_special_tokens(special_tokens)

        self.vocab = self._tokenizer.get_vocab()

    def encode(self, text, out_type=int):
        encoded_ids = self._tokenizer.encode(text).ids
        print(f"Encoded input: {encoded_ids}")
        return encoded_ids

    def decode(self, ids):
        decoded_text = self._tokenizer.decode(ids, skip_special_tokens=True)
        print(f"Decoded output: {decoded_text}")
        return decoded_text

    def token_to_id(self, token):
        return self.vocab.get(token, self.vocab['<unk>'])


def resize_embeddings(model, new_vocab_size):
    old_vocab_size = model.embed.num_embeddings
    if new_vocab_size == old_vocab_size:
        return

    # 调整 embed 层
    old_embeddings = model.embed.weight.data
    new_embeddings = torch.nn.Embedding(new_vocab_size, model.embed.embedding_dim)
    new_embeddings.weight.data[:old_vocab_size, :] = old_embeddings
    # 初始化新增加的嵌入向量
    torch.nn.init.xavier_uniform_(new_embeddings.weight.data[old_vocab_size:])
    model.embed = new_embeddings

    # 调整 fc 层
    old_fc_weight = model.fc.weight.data
    old_fc_bias = model.fc.bias.data
    new_fc = torch.nn.Linear(model.fc.in_features, new_vocab_size)
    new_fc.weight.data[:old_vocab_size, :] = old_fc_weight
    new_fc.bias.data[:old_vocab_size] = old_fc_bias
    # 初始化新增加的输出权重
    torch.nn.init.xavier_uniform_(new_fc.weight.data[old_vocab_size:])
    new_fc.bias.data[old_vocab_size:].fill_(0)
    model.fc = new_fc


def load_model(model_path, tokenizer_path):
    tokenizer = BpeTokenizer(tokenizer_path)
    print(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)

    vocab_size = state_dict['embed.weight'].shape[0]
    print(f"Model vocab size from state_dict: {vocab_size}")

    model = TransformerDecoderModel(vocab_size, embed_size=128, num_heads=2, hidden_dim=256, num_layers=2)

    model.load_state_dict(state_dict, strict=False)

    new_vocab_size = len(tokenizer.vocab)
    if new_vocab_size != vocab_size:
        print(f"Resizing model embeddings from {vocab_size} to {new_vocab_size}")
        resize_embeddings(model, new_vocab_size)

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_len=512, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(input_text, out_type=int)
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    start_token_id = tokenizer.token_to_id("<s>")
    output_ids = torch.tensor([[start_token_id]], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_len):
            if (input_ids >= model.embed.num_embeddings).any() or (output_ids >= model.embed.num_embeddings).any():
                raise ValueError("Token ID out of range in input_ids or output_ids.")

            src_embedding = model.embed(input_ids)
            tgt_embedding = model.embed(output_ids)
            out = model.transformer_decoder(tgt_embedding.transpose(0, 1), src_embedding.transpose(0, 1))
            logits = model.fc(out[-1])

            logits = logits / 1.0  # 调整温度参数
            probs = torch.softmax(logits, dim=-1)

            if top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                probs = probs.masked_fill(indices_to_remove, 0.0)

            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token >= model.embed.num_embeddings:
                next_token = tokenizer.token_to_id("<unk>")

            output_ids = torch.cat((output_ids, torch.tensor([[next_token]], dtype=torch.long)), dim=1)
            if next_token == tokenizer.token_to_id("</s>"):
                break

    response = tokenizer.decode(output_ids[0].tolist())
    return response.strip()


if __name__ == "__main__":
    model_path = 'chat_model.pth'
    tokenizer_path = 'tokenizer.json'

    model, tokenizer = load_model(model_path, tokenizer_path)

    print("Chatbot is online. Type 'exit' to end the conversation.")
    while True:
        input_text = input("User: ")
        if input_text.lower() == 'exit':
            print("Chatbot is offline.")
            break
        response = generate_response(model, tokenizer, input_text, top_k=50, top_p=0.9)
        print(f"Model: {response}")