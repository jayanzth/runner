import pandas as pd
import os
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize

# Set up Flask
app = Flask(__name__)


# Load and preprocess text data
nltk.download('punkt')
stopwords_list = [...]  # Add your stopwords here

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords_list]
    return ' '.join(tokens)

# Load dataset and train the model
df = pd.read_csv('tweet_emotions.csv')
df['cleaned_text'] = df['content'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    cleaned_text = preprocess_text(text)
    text_tfidf = tfidf.transform([cleaned_text])
    prediction = model.predict(text_tfidf)

    return jsonify({"sentiment": prediction[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
