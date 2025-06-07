# -*- coding: utf-8 -*-
"""
MACHINE LEARNING MODEL IMPLEMENTATION

Objective: Create a predictive model using Scikit-learn to classify spam emails.

Deliverable: A Jupyter Notebook showcasing the model's implementation and evaluation.
"""

# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import numpy as np

# --- Configuration ---
# Path to your dataset. You can download a public dataset like 'SMSSpamCollection'
# For example, from Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Make sure to place the 'SMSSpamCollection' file in the same directory as your Jupyter Notebook,
# or provide the full path to the file.
DATASET_PATH = 'SMSSpamCollection'
# DATASET_PATH = 'D:\my projects\codtech projects\internship4\spam.csv'

# --- 2. Load the dataset ---
print(f"Loading dataset from: {DATASET_PATH}")
try:
    # The dataset typically has no header and is tab-separated
    df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['label', 'message'])
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
except FileNotFoundError:
    print(f"Error: The dataset file '{DATASET_PATH}' was not found.")
    print("Please download 'SMSSpamCollection' (e.g., from Kaggle) and place it in the same directory as this notebook, or update the DATASET_PATH variable.")
    # Exit or provide a dummy dataframe to avoid further errors if the file isn't found
    df = pd.DataFrame({'label': ['ham', 'spam', 'ham'], 'message': ['hello', 'win prize', 'meeting']})
    print("Using a small dummy dataset for demonstration. Please fix the path to the real dataset.")
    # You might want to exit here if the dataset is critical for execution
    # import sys
    # sys.exit()


# --- 3. Preprocessing: Convert labels to numerical format (0 for ham, 1 for spam) ---
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nLabels after mapping (0: ham, 1: spam):")
print(df.head())

# --- 4. Split the dataset into training and testing sets ---
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # stratify to maintain label distribution

print(f"\nTraining set size: {len(X_train)} messages")
print(f"Testing set size: {len(X_test)} messages")

# --- 5. Create a Pipeline for Text Vectorization and Model Training ---
# A pipeline streamlines the workflow by allowing you to chain multiple steps together.
# Here, we combine TF-IDF Vectorization with a Multinomial Naive Bayes Classifier.
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)), # TF-IDF Vectorizer
    ('classifier', MultinomialNB()) # Multinomial Naive Bayes Classifier
])

print("\nPipeline created: TF-IDF Vectorizer -> Multinomial Naive Bayes")

# --- 6. Train the model ---
print("\nTraining the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 7. Make predictions on the test set ---
print("\nMaking predictions on the test set...")
y_pred = model_pipeline.predict(X_test)
print("Predictions made.")

# --- 8. Evaluate the model ---
print("\n--- Model Evaluation ---")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham (Predicted)', 'Spam (Predicted)'], yticklabels=['Ham (Actual)', 'Spam (Actual)'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print("\nInterpretation of Confusion Matrix:")
print(f"True Positives (Spam correctly identified): {cm[1, 1]}")
print(f"True Negatives (Ham correctly identified): {cm[0, 0]}")
print(f"False Positives (Ham incorrectly identified as Spam): {cm[0, 1]}")
print(f"False Negatives (Spam incorrectly identified as Ham): {cm[1, 0]}")

# --- 9. Demonstrate with custom examples ---
print("\n--- Demonstrating with Custom Examples ---")

custom_emails = [
    "Congratulations! You've won a free iPhone! Click here to claim your prize.", # Spam
    "Hi, just confirming our meeting for tomorrow at 10 AM. See you then.",        # Ham
    "URGENT! Your bank account has been compromised. Click this link immediately.", # Spam
    "I'll be home late tonight. Can you pick up some groceries?",                  # Ham
    "Free entry to exclusive club for VIP members! Text WIN to 12345 to claim.",   # Spam
    "Hello, how are you doing today? Let's catch up soon.",                       # Ham
    "You have a new message from your friend. Click to view.", # Might be tricky, but often ham
    "Claim your lottery winnings now! Send us your bank details.", # Spam
]

predicted_labels = model_pipeline.predict(custom_emails)
predicted_probabilities = model_pipeline.predict_proba(custom_emails) # Get probabilities

for i, email in enumerate(custom_emails):
    label = "SPAM" if predicted_labels[i] == 1 else "HAM"
    spam_prob = predicted_probabilities[i][1] # Probability of being spam
    print(f"\nEmail: '{email}'")
    print(f"Predicted: {label} (Spam Probability: {spam_prob:.4f})")

print("\n--- End of Program ---")
print("This Jupyter Notebook demonstrates a basic spam email detection model.")
print("You can experiment with different models, feature engineering techniques,")
print("and hyperparameter tuning to improve performance.")