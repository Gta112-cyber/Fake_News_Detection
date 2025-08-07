# app.py
# Flask web interface

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        text = request.form['news']
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)
        result = "Real News" if prediction[0] == 1 else "Fake News"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
