# train_model.py
# Train a fake news detection model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

df = pd.read_csv("dataset/news.csv")
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

X = df['text']
y = df['label']

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save model and vectorizer
with open("models/fake_news_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("models/tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)
