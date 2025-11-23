# LLM-Supervised-Fine-Tuning

A comprehensive repository for supervised fine-tuning of large language models (LLMs) using state-of-the-art techniques and scalable workflows. This project empowers researchers and engineers to efficiently adapt pre-trained LLMs to custom datasets for specialized downstream tasks.

---

## Introduction

LLM-Supervised-Fine-Tuning offers a modular, extensible framework for supervised fine-tuning of transformer-based language models. The repository supports flexible data pipelines, advanced training strategies, and integrated experiment tracking. It is designed for both academic research and production deployment, making model adaptation accessible and reproducible.

---

## Features

- Fine-tune popular LLM architectures (e.g., GPT, LLaMA, Falcon).
- Support for modern datasets in various formats (JSON, CSV, Parquet).
- Scalable training on single and multi-GPU environments.
- Integrated experiment logging with tools like TensorBoard and Weights & Biases.
- Configurable training, evaluation, and inference pipelines.
- Checkpointing, early stopping, and learning rate scheduling.
- Modular codebase for easy customization and extension.

---

## Requirements

Ensure your environment meets the following requirements:

- Python 3.8 or higher
- PyTorch (>=1.10)
- Transformers library (>=4.20)
- Datasets library (>=2.0)
- Additional dependencies:
  - tqdm
  - numpy
  - pandas
  - tensorboard
  - wandb (optional for experiment tracking)

Install requirements via pip:

```bash
pip install -r requirements.txt
```

---

## Installation

Follow these steps to set up the repository:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/NaufalArsa/LLM-Supervised-Fine-Tuning.git
   cd LLM-Supervised-Fine-Tuning
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set Up Weights & Biases:**
   Register at [wandb.ai](https://wandb.ai/) and log in:
   ```bash
   wandb login
   ```

---

## Usage

Fine-tune an LLM on your dataset with the provided training scripts. The typical workflow involves preparing the data, configuring the model, and running the training command.

### 1. Prepare Data

Format your dataset in JSON, CSV, or Parquet. Ensure fields match the model's requirements (e.g., `input`, `output`).

### 2. Configure Training

Edit the `config.yaml` or `train_config.json` file with your model, data, and training parameters.

### 3. Run Training

Command-line example:

```bash
python train.py \
  --config configs/train_config.yaml \
  --data_path data/my_dataset.json \
  --output_dir outputs/my_experiment/
```

### 4. Evaluate and Inference

Evaluate the fine-tuned model:

```bash
python evaluate.py \
  --model_path outputs/my_experiment/checkpoint-1000 \
  --eval_data data/validation.json
```

Run inference on new data:

```bash
python infer.py \
  --model_path outputs/my_experiment/best_model \
  --input_text "Your prompt here"
```

---

## Configuration

Fine-tuning is fully configurable via YAML or JSON files. Typical parameters include:

- `model_name_or_path`: Pre-trained model checkpoint (e.g., `gpt2`, `facebook/llama-7b`)
- `learning_rate`: Set the optimizer learning rate.
- `batch_size`: Choose batch sizes per device.
- `num_epochs`: Total training epochs.
- `logging`: Enable or disable logging to TensorBoard/WandB.
- `save_steps`: Set checkpointing frequency.
- `early_stopping`: Enable and configure early stopping.
- `max_seq_length`: Specify token sequence length.

Sample YAML configuration:

```yaml
model_name_or_path: gpt2
learning_rate: 5e-5
batch_size: 8
num_epochs: 3
logging: true
save_steps: 500
early_stopping: true
max_seq_length: 512
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

- Fork the repository and create a new branch for your feature or bugfix.
- Write clear, concise commit messages.
- Ensure code passes linting and all tests.
- Document new features in the README and code comments.
- Submit a pull request with a detailed description of your changes.

For major changes, open an issue to discuss your proposal first.

---

## License

MIT License

Copyright (c) Naufal Arsa

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---