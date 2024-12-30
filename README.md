This repository contains the implementation of Exercise 1 for the "Deep Learning for NLP" course. The exercise focuses on building, tuning, and evaluating a Recurrent Neural Network (RNN) for sentiment classification using the IMDB dataset.

Overview
The project includes:

Downloading and preprocessing the IMDB dataset.
Building a vocabulary and tokenizing text using the torchtext library.
Implementing a grouped sampler for efficient data loading.
Defining and training a bidirectional LSTM model with PyTorch.
Evaluating the model's performance using F1 scores and other metrics.
Performing hyperparameter optimization and logging results for comparison.
This repository follows the initial template provided for the course. The instructors provided:

Instructions for environment setup and package installation.
Requirements for submission format and naming conventions.
Skeleton code for model training, evaluation, and dataset preprocessing.
My Contributions
Tokenization, vocabulary building, and text-to-index conversion.
Dataset loading with torch.utils.data.Dataset and custom batching.
Definition and training of a bidirectional LSTM model with PyTorch.
Inner training loop, loss calculation, and metric evaluation.
Hyperparameter tuning with early stopping and saving the best model.
The project retains its original structure to align with submission guidelines.

Key Features
Dataset Preparation
Preprocessed the IMDB dataset using slicing to create train, validation, and test splits.
Tokenized text with spaCy and built a vocabulary limited to the top 20,000 tokens.
Bidirectional LSTM Model
Includes an embedding layer, dropout, bidirectional LSTM, and linear layers.
Sequence pooling is applied using mean pooling for dimensionality reduction.
Training and Evaluation
Implemented the training loop with tqdm progress bars.
Tracked key metrics: loss, accuracy, and F1 score.
Integrated regularization to monitor overfitting and saved only the best model.
Hyperparameter Tuning
Compared three configurations of embedding dimensions, RNN sizes, and dropout rates.
Implemented early stopping after three epochs of no improvement.

Here are the main dependencies:
matplotlib==3.7.1
nltk==3.8.1
pandas==1.5.3
scikit-learn==1.2.2
scipy==1.10.1
seaborn==0.12.2
spacy==3.6.1
datasets==3.0.1
torch==2.0.1
torchtext==0.15.2
transformers==4.24.0
evaluate==0.4.0
rouge-score==0.1.2
py7zr

