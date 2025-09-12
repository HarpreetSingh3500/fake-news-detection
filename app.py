import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np

# --- NEW IMPORTS FOR PREPROCESSING ---
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- ONE-TIME NLTK DOWNLOADS ---
# These are needed for lemmatization and stopwords.
# We catch LookupError, which occurs if the data is not found.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# --- 1. INITIALIZE THE APP & PREPROCESSING TOOLS ---
app = Flask(__name__, static_folder='static', template_folder='static')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. LOAD THE ML MODELS ---
vectorizer = None
model = None
try:
    with open("vector.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")


# --- NEW PREPROCESSING FUNCTION ---
# This function replicates the cleaning process from your notebook.
def preprocess_text(text):
    """Cleans and prepares text data for the model."""
    # Remove all non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    review = review.lower()
    # Split into words
    review = review.split()
    # Lemmatize and remove stopwords
    review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
    # Join words back into a single string
    review = ' '.join(review)
    return review

# --- 3. CREATE THE UPGRADED ANALYSIS ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if not vectorizer or not model:
        return jsonify({"error": "ML models not loaded. Check server logs."}), 500

    try:
        text = ""
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.content_type.startswith("text/"):
                text = file.read().decode('utf-8')
            else:
                return jsonify({"error": "Invalid file type. Please upload a .txt file."}), 400
        elif 'text' in request.form:
            text = request.form.get('text')
        
        if not text:
            return jsonify({"error": "No text or file provided for analysis."}), 400

        # --- APPLY THE PREPROCESSING STEP ---
        cleaned_text = preprocess_text(text)

        # --- ML Prediction Logic ---
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction_val = model.predict(vectorized_text)[0]
        prediction_text = "Looking REAL ✔️ News📰" if prediction_val == 1 else "Looking FAKE ⚠ News📰"
        
        raw_score = model.decision_function(vectorized_text)[0]
        confidence = 1 / (1 + np.exp(-abs(raw_score)))
        confidence = (confidence - 0.5) * 2
        confidence = max(0.5, confidence)

        return jsonify({
            "prediction": prediction_text,
            "confidence": confidence,
            "text_length": len(text)
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- 4. SERVE THE FRONTEND ---
@app.route('/')
def serve_index():
    return render_template('index.html')

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

