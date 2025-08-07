# detect.py
# Detect if input news is real or fake

import pickle

model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

def predict_news(news_text):
    vec = vectorizer.transform([news_text])
    prediction = model.predict(vec)
    return "Real News" if prediction[0] == 1 else "Fake News"

if __name__ == "__main__":
    news = input("Enter news text: ")
    print(predict_news(news))
