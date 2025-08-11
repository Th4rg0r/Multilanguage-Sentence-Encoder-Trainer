# Universal Sentence Encoder Trainer

This project provides a complete pipeline to train a high-quality sentence embedding model from scratch on any language. 

## Motivation

Many state-of-the-art NLP models rely on pre-trained weights from large-scale models like BERT or T5. However, these models are not available for all languages, especially those that are less common or low-resource. 

This project was created to solve that problem. It allows you to build a powerful sentence encoder using a Transformer-based architecture, learning entirely from your own text data. The end result is a model that can produce meaningful vector representations (embeddings) of sentences, which can be used for tasks like semantic search, clustering, and sentence similarity in any language.

## Core Technologies

This project is built on a modern, state-of-the-art stack for NLP tasks, chosen for its power, flexibility, and strong community support.

- **Python**: The universal language for machine learning.
- **PyTorch**: A leading deep learning framework that provides flexibility and a dynamic computation graph, making it ideal for complex models and research.
- **Transformer Encoder Architecture**: The model is based on the same fundamental building block used in groundbreaking models like BERT and GPT. This is the de-facto standard for high performance in NLP.
- **Hugging Face `tokenizers`**: We use this library to train a custom, fast, and efficient Byte-Pair Encoding (BPE) tokenizer directly on your data.
- **Optuna**: A powerful, state-of-the-art hyperparameter optimization framework. It automates the tedious process of finding the best model architecture and learning parameters, using efficient sampling and pruning algorithms.
- **YAML**: All configuration is managed through a single, human-readable `config.yaml` file, separating the logic of the code from the experimental parameters.

## Workflow Overview

The project supports a full end-to-end workflow:

1.  **Data Preparation**: Automatically splits your raw text into training and test sets.
2.  **Tokenizer Training**: Trains a custom tokenizer on your data.
3.  **Hyperparameter Optimization (Optional but Recommended)**: Uses Optuna to find the best model architecture (e.g., number of layers, embedding size) and training parameters for your specific dataset.
4.  **Pre-training**: The model first learns the structure of the language using a Masked Language Modeling (MLM) objective.
5.  **Fine-tuning**: The pre-trained model is then fine-tuned using a Contrastive Learning (InfoNCE loss) objective. This step is crucial for producing high-quality sentence embeddings that are well-suited for semantic similarity tasks.

## Installation and Usage

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# 2. Create a Python virtual environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install the required packages
pip install -r requirements.txt
```

### How to Use

Follow these steps to train your own model.

**Step 1: Prepare Your Data**

Place your raw text data into a single file named `input.txt` in the root of the project directory. The file should have **one sentence per line**.

**Step 2: Initial Setup**

First, you need to process your data and create a tokenizer. Run the following commands in order:

```bash
# This splits your input.txt into training and test sets
python train.py --setup_data

# This trains a new tokenizer on your training data
python train.py --setup_tokenizer
```

**Step 3: Train the Model**

You have two options:

**Option A: Find the Best Hyperparameters (Recommended)**

This will use Optuna to automatically find the best model configuration for your data. This can take a long time, but typically yields the best results.

```bash
python train.py --optimize
```

**Option B: Train with a Specific Configuration**

This will use a specific set of hyperparameters to pre-train the model.

```bash
python train.py
```

By default, the script will automatically load the **best hyperparameters** found by the last `--optimize` run. If you want to ignore the optimization results and use the specific parameters written in the `config.yaml` file, set `override_with_custom_params.enabled` to `true` in the `train` section of the config.

**Step 4: Fine-tune the Model**

After the pre-training stage is complete, you must fine-tune the model to produce high-quality sentence embeddings.

```bash
python train.py --finetune
```

### Starting Fresh (`--new` flag)

By default, the script will always try to resume from the last saved state. If you want to start a process from scratch, use the `--new` flag.

-   `python train.py --new`: Ignores any saved pre-trained model and starts training from scratch.
-   `python train.py --optimize --new`: Deletes the old optimization study and starts a new one.
-   `python train.py --finetune --new`: Ignores any previously fine-tuned model and starts a new fine-tuning session from the pre-trained model.

## Configuration (`config.yaml`)

The `config.yaml` file is the central control panel for the entire project. Here are some of the most important settings:

-   `data.test_size`: The proportion (e.g., `0.1` for 10%) of your dataset to hold out for testing.
-   `tokenizer.vocab_size`: The total number of unique tokens you want in your tokenizer's vocabulary. A larger size captures more words but increases model size.
-   `tokenizer.keep_punctuation`: A boolean (`true` or `false`) to control whether punctuation is stripped from the text or kept as separate tokens.
-   `optimize.n_trials`: The total number of different hyperparameter combinations Optuna should test.
-   `train.override_with_custom_params.enabled`: Set this to `true` to ignore the results from Optuna and instead use the specific model parameters defined under this section for pre-training.
-   `finetune.freeze_layer_ratio`: A float between 0.0 and 1.0. This controls what percentage of the model's **bottom layers** are frozen during fine-tuning. For example, `0.25` freezes the first 25% of the layers, preserving their general knowledge.
-   `finetune.scheduler_patience`: The number of epochs with no improvement on the validation loss before the learning rate is reduced.

## Final Output

The trained models are saved in the `models/` directory:
-   `models/model.pt`: The base model from the pre-training step.
-   `models/model_finetuned.pt`: The final, fine-tuned model, ready for producing sentence embeddings.
