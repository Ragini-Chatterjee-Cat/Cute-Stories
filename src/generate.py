"""
Script for generating text using a trained model.
"""
import torch
import argparse

from data.dataset import load_tiny_stories
from models.transformer import TransformerDecoderOnly
from utils.config import *


def generate_story(model, enc, device, prompt='\n', max_tokens=MAX_NEW_TOKENS):
    """
    Generate text from the trained model.

    Args:
        model: Trained transformer model
        enc: Tiktoken encoder
        device: Device to run on
        prompt: Starting prompt for generation
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated text string
    """
    model.eval()
    context = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated_ids = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
        story = enc.decode(generated_ids)

    return story


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate stories using trained Transformer model')
    parser.add_argument('--model_path', type=str, default=MODEL_SAVE_PATH,
                       help='Path to trained model weights')
    parser.add_argument('--prompt', type=str, default='\n',
                       help='Starting prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=MAX_NEW_TOKENS,
                       help='Maximum number of tokens to generate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset to get encoder and vocab size
    print("Loading encoder...")
    _, _, enc = load_tiny_stories()
    vocab_size = enc.n_vocab

    # Initialize model
    print("Initializing model...")
    model = TransformerDecoderOnly(vocab_size, N_EMBD, BLOCK_SIZE, num_layers=N_LAYER, n_heads=N_HEAD)
    model = model.to(device)

    # Load trained weights
    print(f"Loading model weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Generate and print story
    print("\nGenerating story...")
    story = generate_story(model, enc, device, args.prompt, args.max_tokens)
    print("\nGenerated Story:")
    print("=" * 80)
    print(story)
    print("=" * 80)


if __name__ == '__main__':
    main()
