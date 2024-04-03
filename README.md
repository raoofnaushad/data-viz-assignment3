
# Text Classification Project README

## Overview

This project focuses on training, saving, and evaluating machine learning models for text classification. It utilizes three different models to predict categories of text data accurately. The project is divided into two main Jupyter notebooks:

1. **Training Models.ipynb**: This notebook is responsible for training three distinct models on a text classification task. It preprocesses the data, trains each model, and saves them to disk for later use.

2. **Raoof_Naushad_Inference.ipynb**: This notebook loads the previously trained and saved models to perform inference on new, unseen test data. It evaluates each model's performance using classification reports and accuracy metrics.

## Models

The project explores the following machine learning models for text classification:

1. **Multinomial Naive Bayes (MNB)**: A popular choice for text classification tasks, especially suitable for datasets with discrete features (e.g., word counts or TF-IDF vectors).

2. **Random Forest Classifier (RFC)**: An ensemble learning method for classification that operates by constructing a multitude of decision trees at training time.

3. **Logistic Regression (LR)**: Despite its name, logistic regression is a linear model for binary classification that can be extended to multiclass classification.

## Usage

### Training the Models

1. Open and run the `Training Models.ipynb` notebook.
2. The notebook will guide you through loading the dataset, preprocessing the text data, training each model, and saving them to disk.
3. Saved models will be stored in the same directory as the notebook, with filenames indicating the model type.

### Performing Inference and Evaluation

1. Ensure that the models are trained and saved as described above.
2. Open the `Raoof_Naushad_Inference.ipynb` notebook.
3. This notebook will automatically load the saved models and apply them to the test data.
4. The evaluation metrics, including accuracy and the classification report for each model, will be displayed.

## Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- scikit-learn
- NLTK or similar for text preprocessing
- joblib for saving and loading models
- Optuna for hyperparameter optimization (optional)

Make sure to install all necessary libraries using pip:

```bash
pip install numpy scipy scikit-learn nltk joblib optuna
```

## Dataset

The dataset used for training and testing should be text data labeled with categories. For training purposes, this project assumes a dataset split into training and test sets.

## Contributions

This project is open to contributions. If you have suggestions for improving the models, preprocessing steps, or any other part of the project, feel free to create a pull request.

