
# Arabic Text Classification Models

This repository contains Python notebooks implementing text classification models for Arabic reviews using different machine learning algorithms. The notebooks preprocess text data, convert it into vector representations, and train classifiers to predict sentiment labels for reviews.

## Notebooks

### Traditional Machine Learning Approach
- **[Traditional_ML_Model.ipynb](Traditional_ML_Model.ipynb)**: This notebook implements a traditional machine learning approach for text classification. It preprocesses Arabic text data, converts it into vector representations using Word2Vec embeddings, and trains multiple classifiers (e.g., SGD Classifier, Random Forest Classifier, LinearSVC) to predict sentiment labels for reviews.

### Recurrent Neural Network (RNN) Approach
- **[RNN_Model.ipynb](RNN_Model.ipynb)**: This notebook implements a Recurrent Neural Network (RNN) approach for text classification. It preprocesses Arabic text data, tokenizes and pads sequences, constructs an LSTM-based neural network, and trains the model to predict sentiment labels for reviews.

## Dataset

The dataset used for training and evaluation is stored in `Dataset.tsv`. It contains Arabic text reviews along with sentiment labels (Negative, Positive, Mixed).

## Dependencies

Both notebooks require the following dependencies to be installed:

- numpy
- scikit-learn
- nltk
- seaborn
- matplotlib
- plotly
- gensim
- tensorflow
- keras

You can install these dependencies using `pip install`.

## Usage

1. Ensure that you have installed all the required dependencies.
2. Download the dataset (`Dataset.tsv`).
3. Choose the notebook corresponding to the desired approach (Traditional ML or RNN).
4. Run the notebook.

## Evaluation

Each notebook evaluates the performance of the classification model using various metrics, including accuracy, precision, recall, F1-score, and confusion matrix.

## Note

- Ensure that you have enough computational resources to load the pre-trained Word2Vec model (`full_grams_cbow_300_twitter.mdl`) and train the RNN model as they may require substantial memory and computational power.

