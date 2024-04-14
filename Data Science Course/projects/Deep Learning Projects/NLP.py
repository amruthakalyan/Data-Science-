import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load email dataset (you can replace this with your own dataset)
# The dataset should have two columns: "text" containing email content and "label" indicating spam or not spam
emails = pd.read_csv("spam_dataset.csv")

# Preprocess email text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())
    # Remove stop words and non-alphabetic characters
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_tokens)

emails['text_processed'] = emails['text'].apply(preprocess_text)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(emails['text_processed'], emails['label'], test_size=0.2, random_state=42)

# Convert text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Predict on test set
y_pred = clf.predict(X_test_vectorized)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
