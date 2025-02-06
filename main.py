from dataset import load_tiny_stories
from decoder import TransformerDecoderOnly
import torch

# Function to generate text from the trained model
def generate_story(model, enc, device):
    context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)
    story = enc.decode(model.generate(context, max_new_tokens=200)[0].tolist())
    return story

if __name__ == '__main__':
    # Load dataset
    train_data, val_data, enc = load_tiny_stories()
    vocab_size = enc.vocab_size

    # Initialize model
    model = TransformerDecoderOnly(vocab_size, 512, 64)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained weights
    model.load_state_dict(torch.load("transformer_decoder.pth", map_location=model.device))
    model.eval()  # Set model to evaluation mode

    # Generate and print a story
    print("Generated Story:")
    print(generate_story(model, enc, model.device))
