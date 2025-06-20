# SMS Spam Classifier (RNN + CNN)

## Overview:
This project implements a hybrid deep learning model combining **RNN (LSTM)** and **CNN** to classify SMS messages as **Spam** or **Not Spam (Ham)**. The model achieves **99.03% accuracy** on the test set.

## Features:
- **Preprocessing**: Tokenization, padding, and label encoding
- **Hybrid Architecture**:  
  - Embedding Layer  
  - 1D Convolutional Layer  
  - Max Pooling  
  - Two LSTM Layers  
  - Dense Layers with Dropout  
- **Early Stopping**: Prevents overfitting
- **Saved Model**: Includes `spam_classifier.keras` and `tokenizer.pkl` for reuse

## Requirements:
- numpy
- pandas
- matplotlib
- scikit-learn
- keras
- tensorflow

## Usage:
### 1. Load the Model and Tokenizer:
- from keras.models import load_model
- import pickle
- model = load_model("spam_classifier.keras")
- with open("tokenizer.pkl", "rb") as handle:
    - tokenizer = pickle.load(handle)

### 2. Predict Spam/Ham:
- from tensorflow.keras.preprocessing.sequence import pad_sequences
- def predict_spam(message):
   - sequence = tokenizer.texts_to_sequences([message])
   - padded_sequence = pad_sequences(sequence, maxlen=100)
   - prediction = model.predict(padded_sequence)
   - return "Spam" if prediction[0][0] > 0.5 else "Not Spam"

# Example:
- print(predict_spam("WIN A FREE PRIZE NOW!")) [Output: Spam]

### 3. Interactive Input:
- while True:
    - user_input = input("Enter message (or 'quit' to exit): ")
    - if user_input.lower() == "quit":
       - break
    - print("Prediction:", predict_spam(user_input))

## Dataset:
- **Source**: Custom CSV (sample_texts.csv)  
- **Columns**:  
  - label: 0 (Ham) / 1 (Spam)  
  - message: Raw SMS text  

## Training:
- **Epochs**: 10 (with early stopping)  
- **Batch Size**: 32  
- **Validation Accuracy**: 98.65%  

## Files:
- spam_classifier_rnn_cnn.ipynb: Full training/evaluation notebook  
- spam_classifier.keras: Saved model  
- tokenizer.pkl: Saved tokenizer  

## Performance:
Test Accuracy: 99.03%
