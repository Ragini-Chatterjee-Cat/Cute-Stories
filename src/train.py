"""
Training script for the Transformer Decoder model.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data.dataset import load_tiny_stories, get_batch
from models.transformer import TransformerDecoderOnly
from utils.config import *


def loss_function(logits, targets):
    """
    Calculate cross-entropy loss.

    Args:
        logits: Model output logits
        targets: Target token IDs

    Returns:
        Loss value
    """
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def evaluate(model, train_data, val_data, batch_size, block_size, device, eval_iters):
    """
    Evaluate the model on train and validation sets.

    Args:
        model: The transformer model
        train_data: Training data tensor
        val_data: Validation data tensor
        batch_size: Batch size
        block_size: Sequence length
        device: Device to run evaluation on
        eval_iters: Number of evaluation iterations

    Returns:
        dict: Dictionary with 'train' and 'val' losses
    """
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

    model.train()
    return losses


def train():
    """Main training function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    train_data, val_data, enc = load_tiny_stories(NUM_SAMPLES, TRAIN_SPLIT)
    vocab_size = enc.n_vocab
    print(f"Vocabulary size: {vocab_size}")

    # Initialize model
    print("Initializing model...")
    model = TransformerDecoderOnly(
        vocab_size,
        N_EMBD,
        BLOCK_SIZE,
        num_layers=N_LAYER,
        n_heads=N_HEAD
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Training loop
    print("Starting training...")
    model.train()

    for iter in tqdm(range(MAX_ITERS)):
        # Evaluation
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = evaluate(
                model, train_data, val_data,
                BATCH_SIZE, BLOCK_SIZE, device, EVAL_ITERS
            )
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Training step
        X, Y = get_batch('train', train_data, val_data, BATCH_SIZE, BLOCK_SIZE, device)
        logits = model(X)
        loss = loss_function(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Save the trained model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training complete!")

    # Generate sample text
    print("\nGenerating sample story...")
    model.eval()
    context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated_text = enc.decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist())
    print("\nGenerated Story:")
    print(generated_text)


if __name__ == '__main__':
    train()
