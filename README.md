# Cute Stories Transformer Decoder

This repository contains a project for training and using a Transformer Decoder model to generate text. The project is built with PyTorch and leverages GPT-2 tokenization for training on datasets like TinyStories. A FastAPI-based API is also included for generating text using the trained model.

---

## Features

- **Custom Transformer Decoder:** Implements a custom Transformer Decoder architecture from scratch using PyTorch.
- **Dataset Management:** A script to preprocess and tokenize the dataset for training.
- **Training Script:** Code to train the Transformer model with adjustable hyperparameters.
- **API Integration:** A FastAPI-based API for generating text from a prompt using the trained model.

---

## File Structure

```plaintext
├── dataset.py          # Handles dataset loading and tokenization
├── decoder.py          # Transformer Decoder implementation
├── main.py             # API server using FastAPI
├── model.py            # Defines the Transformer model
├── training.py         # Trains the transformer model        
├── requirements.txt    # Python dependencies for the project
├── README.md           # Project documentation
