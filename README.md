# Transformer-Based Text Generation API

## Overview
This project implements a text generation API using a Transformer-based decoder-only model. It is built using FastAPI and PyTorch, and it employs a GPT-2 tokenizer for text input processing. The API serves a trained model and allows users to generate text based on a given prompt.

## Features
- Transformer-based text generation
- FastAPI for serving the model as an API
- Pretrained GPT-2 tokenizer for encoding input text
- RESTful API endpoint for text generation

## Installation
To set up the project, first install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure you have `cmake` installed, as it is required for dependencies like `pyarrow`:

```bash
sudo apt-get install cmake  # For Ubuntu/Debian
brew install cmake          # For macOS
```

## Usage

### Running the API
Start the FastAPI server using Uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoint
- **Generate Text**: `GET /generate/`
  - **Parameters:**
    - `prompt` (str, required): The input text prompt.
    - `max_length` (int, optional, default=100): Maximum number of tokens to generate.
  - **Example Request:**
    ```
    http://localhost:8000/generate/?prompt=Once%20upon%20a%20time&max_length=50
    ```
  - **Example Response:**
    ```json
    {
        "prompt": "Once upon a time",
        "generated_text": "Once upon a time, in a faraway land, there was a kingdom..."
    }
    ```

## Model
The model is a transformer-based decoder-only model implemented in PyTorch. It is loaded from `Model.pth`. Ensure that the trained model weights are saved in the project directory before running the API.

## File Structure
```
├── api.py                # FastAPI implementation
├── model.py              # Transformer model definition
├── train.py              # Model training script
├── dataset.py            # Dataset processing (if applicable)
├── config.py             # Configuration parameters
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
```

## Deploying to GitHub
1. Initialize a Git repository:
   ```bash
   git init
   ```
2. Add files to the repository:
   ```bash
   git add .
   ```
3. Commit changes:
   ```bash
   git commit -m "Initial commit with API and model implementation"
   ```
4. Create a GitHub repository and link it:
   ```bash
   git remote add origin https://github.com/your-username/your-repo.git
   ```
5. Push to GitHub:
   ```bash
   git branch -M main
   git push -u origin main
   ```

## Future Improvements
- Implement a training pipeline for fine-tuning the model.
- Improve tokenization and support additional languages.
- Deploy using Docker and cloud services.

## License
This project is open-source and available under the MIT License.

