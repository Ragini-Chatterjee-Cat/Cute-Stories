"""
Configuration parameters for model training and inference.
"""

# Model hyperparameters
BATCH_SIZE = 128
BLOCK_SIZE = 64
LEARNING_RATE = 3e-4
N_EMBD = 512
N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1

# Training parameters
MAX_ITERS = 10000
EVAL_INTERVAL = 500
EVAL_ITERS = 100

# Dataset parameters
NUM_SAMPLES = 12000
TRAIN_SPLIT = 0.9

# Generation parameters
MAX_NEW_TOKENS = 200

# Model save path
MODEL_SAVE_PATH = "transformer_decoder.pth"
