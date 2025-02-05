from datasets import load_dataset
import torch
import tiktoken

def load_tiny_stories():
    dataset = load_dataset('roneneldan/TinyStories')
    text = '\n'.join(dataset['train']['text'][:12000])
    enc = tiktoken.get_encoding('gpt2')
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], enc

def get_batch(split, train_data, val_data, batch_size, block_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)