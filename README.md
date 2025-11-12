# Cute Stories - Transformer Decoder

A PyTorch implementation of a decoder-only Transformer model for generating creative short stories. This project uses the TinyStories dataset and GPT-2 tokenization to train a custom text generation model from scratch.

## Features

- **Custom Transformer Architecture**: Implements a decoder-only Transformer with multi-head attention from scratch
- **Modular Design**: Clean, organized codebase with separate modules for models, data handling, and utilities
- **Flexible Training**: Configurable hyperparameters and training settings
- **Text Generation**: Generate stories from custom prompts or random initialization
- **Well-Documented**: Comprehensive docstrings and type hints throughout

## Project Structure

```
Cute-Stories/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py       # Transformer model components
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Dataset loading and batch generation
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py            # Configuration parameters
│   ├── train.py                 # Training script
│   └── generate.py              # Generation script
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Cute-Stories.git
cd Cute-Stories
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the model using the default configuration:

```bash
python -m src.train
```

This will:
- Load the TinyStories dataset
- Initialize the Transformer model
- Train for the specified number of iterations
- Save the trained model to `transformer_decoder.pth`
- Generate a sample story

### Generating Stories

Generate stories using a trained model:

```bash
# Generate with default prompt
python -m src.generate

# Generate with custom prompt
python -m src.generate --prompt "Once upon a time" --max_tokens 300

# Use a different model checkpoint
python -m src.generate --model_path path/to/model.pth
```

## Configuration

Modify hyperparameters in `src/utils/config.py`:

```python
# Model hyperparameters
BATCH_SIZE = 128        # Training batch size
BLOCK_SIZE = 64         # Sequence length
LEARNING_RATE = 3e-4    # Learning rate
N_EMBD = 512           # Embedding dimension
N_HEAD = 8             # Number of attention heads
N_LAYER = 6            # Number of transformer layers

# Training parameters
MAX_ITERS = 10000      # Training iterations
EVAL_INTERVAL = 500    # Evaluation frequency

# Generation parameters
MAX_NEW_TOKENS = 200   # Maximum tokens to generate
```

## Model Architecture

The model consists of:

1. **Positional Embedding Layer**: Adds positional information to token embeddings
2. **Multi-Head Attention**: Captures relationships between tokens
3. **Transformer Blocks**: 6 stacked layers with attention and feed-forward networks
4. **Layer Normalization**: Stabilizes training
5. **Output Layer**: Projects to vocabulary size for next token prediction

### Key Components

- **PositionalEmbedding**: Sinusoidal positional encoding
- **MultiHeadAttention**: Self-attention with 8 heads
- **TransformerBlock**: Attention + feed-forward with residual connections
- **TransformerDecoderOnly**: Complete decoder model with autoregressive generation

## Dataset

Uses the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) from Hugging Face, which contains synthetic short stories suitable for language model training.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- datasets
- tiktoken
- tqdm

See `requirements.txt` for complete dependencies.

## Example Output

```
Generated Story:
Once upon a time, there was a little girl named Lily. She loved to play with her toys.
One day, she found a big box. Inside the box was a beautiful doll. Lily was so happy!
She played with the doll all day long. The end.
```

## Training Details

- **Optimizer**: AdamW with learning rate 3e-4
- **Learning Rate Schedule**: Step decay (gamma=0.5 every 2000 steps)
- **Loss Function**: Cross-entropy loss
- **Training Duration**: ~10,000 iterations
- **GPU Support**: Automatically uses CUDA if available

## License

MIT License - feel free to use this code for your own projects!

## Acknowledgments

- Built with PyTorch
- Uses the TinyStories dataset by Eldan & Li
- Tokenization via tiktoken (GPT-2 encoding)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Author

Ragini Chatterjee

---

**Note**: This is an educational project demonstrating Transformer architecture implementation. The generated stories are simple and suitable for learning purposes.
