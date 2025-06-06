# MIE1517-Project-4 Sentiment Analysis of Movie Reviews

This project implements and compares two deep learning models for classifying IMDb movie reviews as **positive** or **negative**:
- A custom-built **LSTM-based RNN**
- A **BERT transformer model** using Hugging Face's `bert-base-uncased`

Run this notebook directly in **Google Colab** (recommended).
> [Open in Colab](# http://https://colab.research.google.com/github/Fulankeee/MIE1517-Project-4/blob/main/A4.ipynb#scrollTo=rk7aDAaR2_wz)
---

## Part A – Word-Level RNN Model

### Dataset
- Dataset: [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- Reviews: 50,000 labeled samples
- Preprocessing: Lowercasing, punctuation removal, tokenization, and word-to-index mapping

### Model Architecture

```python
Embedding to LSTM to FullyConnected to Sigmoid
```
- Embedding size: 300
- LSTM hidden units: 128
- Output: Single sigmoid neuron for binary classification

### Training Details
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam
- Epochs: 5
- Batch size: 64

### RNN Accuracy
- **Train Accuracy**: 88%
- **Validation Accuracy**: 85%
- Handles long-term dependencies moderately well
---

## Part B – Transformer-Based BERT Classifier

### Model Setup
- Pretrained model: `bert-base-uncased`
- Tokenizer: `BertTokenizer`
- Classifier: `BertForSequenceClassification`

### Training Strategy
- Input: tokenized text + attention masks
- Learning Rate: 0.00001
- Epochs: 3

### BERT Accuracy
- **Validation Accuracy**: **92.5%**
- Faster convergence and better contextual understanding than RNN
- Robust to mixed sentiment and longer sequences

## Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| RNN   | 85%     | Lightweight, word-based, slower convergence |
| BERT  | **92.5%**| Context-aware, fine-tuned, state-of-the-art performance |

---

## Discussion
- RNNs performs decently but struggle with sarcasm and context.
- BERT clearly outperforms due to its ability to capture semantic meaning.
- Preprocessing and padding/tokenization play a critical role in both pipelines.

---
