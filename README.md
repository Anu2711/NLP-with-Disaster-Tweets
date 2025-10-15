# NLP with Disaster Tweets 🚨🐦

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-red.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.82-success.svg)

A natural language processing project that classifies tweets as disaster-related or not, leveraging DistilBERT and multi-GPU training.

## 🎯 Project Overview

This project tackles the [Kaggle NLP Getting Started competition](https://www.kaggle.com/competitions/nlp-getting-started/overview), which challenges participants to build a model that can identify whether a tweet is about a real disaster or not. 

**The Challenge**: Tweets are notoriously difficult to parse—they're short, filled with slang, use inconsistent grammar, and often include sarcasm. A tweet saying "California is on fire!" could be literal (a wildfire) or figurative (great weather). This project explores how transformer-based models handle this ambiguity.

## 📚 Learning Journey

This project was undertaken while working through the **[Python Natural Language Processing Cookbook, Second Edition](https://www.packtpub.com/product/python-natural-language-processing-cookbook-second-edition/9781838987312)**. My goals were to:

1. **Master NLP fundamentals**: Deep dive into tokenization, embeddings, and modern NLP architectures
2. **Work with real Kaggle data**: Experience the messiness of real-world text classification
3. **Explore transformers**: Implement and fine-tune pre-trained models like DistilBERT
4. **Optimize training**: Learn to parallelize training across multiple GPUs for faster experimentation

## ✨ Key Features

- **DistilBERT Architecture**: Leverages a distilled version of BERT for efficient text classification
- **Multi-GPU Training**: Parallelized across 2 T4 GPUs (courtesy of Kaggle) for faster iteration
- **Hyperparameter Tuning**: 5-fold cross-validation to find optimal model configuration
- **Competition Performance**: Achieved F1-score of **0.82** on the competition leaderboard
- **End-to-End Pipeline**: From data preprocessing to model deployment and submission generation

## 🔬 Technical Approach

### 1. Data Preprocessing

The dataset consists of tweets labeled as disaster (1) or not disaster (0). Preprocessing steps include:

- **Text cleaning**: Handling URLs, mentions, hashtags, and special characters
- **Tokenization**: Using DistilBERT's tokenizer to convert text into model-ready format
- **Handling imbalance**: Analyzing class distribution and applying appropriate techniques

### 2. Model Architecture

**DistilBERT** was chosen for several reasons:
- 40% smaller than BERT while retaining 97% of its performance
- Faster inference time—crucial for real-time disaster detection systems
- Pre-trained on a massive corpus, providing strong language understanding
- Easy to fine-tune on domain-specific data

**Model Configuration**:
```
Base Model: DistilBERT (distilbert-base-uncased)
Classification Head: Linear layer for binary classification
Max Sequence Length: 128 tokens
Dropout: 0.1 for regularization
```

### 3. Training Strategy

**Hyperparameter Tuning with 5-Fold Cross-Validation**:
- Learning rates: [2e-5, 3e-5, 5e-5]
- Batch sizes: [16, 32]
- Epochs: [3, 4, 5]

**Multi-GPU Parallelization**:
Leveraged Kaggle's 2 T4 GPUs to:
- Run multiple hyperparameter combinations simultaneously
- Reduce total training time by ~50%
- Enable more extensive experimentation within time constraints

**Optimization**:
- Optimizer: AdamW with weight decay
- Learning rate scheduler: Linear warmup followed by decay
- Loss function: Binary cross-entropy

### 4. Evaluation

The competition evaluates submissions using the **F1-score**, which balances precision and recall—critical for disaster detection where both false positives and false negatives have consequences.

**Final Results**:
- **F1-Score**: 0.82
- **Cross-validation performance**: Consistent across all folds, indicating stable model

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained DistilBERT model
- **Scikit-learn**: Cross-validation and metrics
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Kaggle API**: Dataset access and submission
- **CUDA**: GPU acceleration

## 🔮 Future Improvements

- [ ] **Ensemble methods**: Combine DistilBERT with other transformer models (RoBERTa, ALBERT)
- [ ] **Error analysis**: Deep dive into misclassified tweets to identify patterns
- [ ] **Deployment**: Create REST API for real-time disaster tweet detection
