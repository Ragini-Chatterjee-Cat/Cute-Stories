import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import load_tiny_stories, get_batch
from decoder import TransformerDecoderOnly
from model import PositionalEmbedding
from tqdm import tqdm

# Hyperparameters
batch_size = 128
block_size = 64
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 6
dropout = 0.1
max_iters = 10000
eval_interval = 500

# Load dataset
train_data, val_data, enc = load_tiny_stories()
vocab_size = enc.vocab_size

# Initialize model
model = TransformerDecoderOnly(vocab_size, n_embd, block_size, num_layers=n_layer, n_heads=n_head)
model = model.to(device)

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

def loss_function(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Training loop
model.train()
for iter in tqdm(range(max_iters)):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        losses = {'train': 0, 'val': 0}
        with torch.no_grad():
            for split in ['train', 'val']:
                for _ in range(eval_iters):
                    X, Y = get_batch(split, train_data, val_data, batch_size, block_size, device)
                    logits = model(X)
                    loss = loss_function(logits, Y)
                    losses[split] += loss.item()
                losses[split] /= eval_iters
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        model.train()
    
    X, Y = get_batch('train', train_data, val_data, batch_size, block_size, device)
    logits = model(X)
    loss = loss_function(logits, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Save the trained model
torch.save(model.state_dict(), "transformer_decoder.pth")

# Generate text using the trained model
def generate_story(model, enc, device):
    model.eval()
    context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated_text = enc.decode(model.generate(context, max_new_tokens=200)[0].tolist())
    return generated_text

print("Generated Story:")
print(generate_story(model, enc, device))
