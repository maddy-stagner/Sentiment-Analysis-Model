# Sentiment-Analysis-Model
A simple sentiment classification model using TF-IDF vectorization and Logistic Regression on Twitter data. Trained on train split only, with hyperparameter tuning (C and max_iter) via GridSearchCV. Achieves ~79% accuracy on test set. Built with scikit-learn and pandas
# Sentiment Analysis with TF-IDF and Logistic Regression

This repository contains a Jupyter notebook for building a binary sentiment classification model on Twitter data. The model uses TF-IDF for feature extraction (fitted only on train data) and Logistic Regression with hyperparameter tuning.

## Dataset
- Uses the Sentiment140 dataset (1.6M tweets) from `training.1600000.processed.noemoticon.csv`.
- Download it [here](https://www.kaggle.com/datasets/kazanova/sentiment140) and place in the notebook directory.

## Features
- Data preprocessing and splitting (train: 60%, val: 20%, test: 20%).
- TF-IDF vectorization with max_features=5000.
- Hyperparameter tuning: C=[0.1, 1, 5], max_iter=[2000] via GridSearchCV.
- Evaluation: Classification report on test data (~79% accuracy).

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the notebook: `jupyter notebook 1st_nlp_practice.ipynb`

## Results
Best params: {'C': 1, 'max_iter': 2000}
Test performance:
              precision    recall  f1-score   support
           0       0.80      0.78      0.79    159790
           4       0.79      0.80      0.79    160210
    accuracy                           0.79    320000

