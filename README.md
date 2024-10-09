# Transformers Text Classification

This repository is part of a deep learning project focused on sentiment analysis of Vietnamese text reviews using Transformer Encoder architecture. The implementation leverages the powerful self-attention mechanisms of Transformers to classify sentiments as positive or negative accurately.

## Project Overview

The project aims to tackle the challenges of sentiment analysis in the Vietnamese language by employing a Transformer Encoder model. This approach emphasizes model optimization and architectural efficacy to enhance the accuracy of sentiment classification.

## Features

- **Text Preprocessing** - (data_processing.py and tokenizer.py): Implements comprehensive preprocessing of Vietnamese text data.
- **Transformer Model** - (model.py): Utilizes Transformer Encoder layers for effective sentiment analysis.
- **Hyperparameter Tuning** - (train.py): Details systematic tuning to find the optimal model configuration.
- **Performance Evaluation** - (train.py): Demonstrates model's capabilities with accuracy metrics on validation and test datasets.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/tanvu10/Transformers-text-classification.git
cd Transformers-text-classification
pip install -r requirements.txt
```

### Usage
Run the training script with:

```bash
python train.py
```

## Results
The model achieved an accuracy of 85.88% on the validation set and 84.53% on the test set, showcasing its effectiveness in handling the subtleties of Vietnamese text sentiment classification.

## Documentation
For more details on the architecture and usage, refer to the inline comments and docs folder.