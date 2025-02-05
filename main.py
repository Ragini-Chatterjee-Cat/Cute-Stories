from dataset import load_tiny_stories, get_batch
from decoder import TransformerDecoderOnly
import torch

def generate_story(model, enc, device):
    context = torch.tensor(enc.encode('\n'), dtype=torch.long, device=device).unsqueeze(0)
    story = enc.decode(model.generate(context, max_new_tokens=200)[0].tolist())
    return story

if __name__ == '__main__':
    train_data, val_data, enc = load_tiny_stories()
    vocab_size = enc.n_vocab
    model = TransformerDecoderOnly(vocab_size, 512, 64).to('cuda' if torch.cuda.is_available() else 'cpu')
    print("Generated Story:")
    print(generate_story(model, enc, 'cuda' if torch.cuda.is_available() else 'cpu'))
