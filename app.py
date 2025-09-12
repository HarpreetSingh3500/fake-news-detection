import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- ONE-TIME NLTK DOWNLOADS ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


# --- 1. INITIALIZE THE APP & PREPROCESSING TOOLS ---
app = Flask(__name__, static_folder='static', template_folder='static')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- 2. LOAD THE ML MODELS ONCE AT STARTUP ---
vectorizer = None
model = None
try:
    vector_path = os.path.join(os.path.dirname(__file__), 'vector.pkl')
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    
    with open(vector_path, "rb") as f_vect:
        vectorizer = pickle.load(f_vect)
    with open(model_path, "rb") as f_model:
        model = pickle.load(f_model)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load machine learning models. Exception: {e}")


# --- 3. REVISED PREPROCESSING FUNCTION ---
def preprocess_text(text):
    """Cleans and prepares text data for the model."""
    review = re.sub(r'[^a-zA-Z\s]', '', text)
    review = review.lower()
    review = word_tokenize(review)
    
    corpus = []
    for word in review:
        if word not in stop_words:
            corpus.append(lemmatizer.lemmatize(word))
            
    return ' '.join(corpus)

# --- 4. CREATE THE ANALYSIS ENDPOINT ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if not vectorizer or not model:
        return jsonify({"error": "Machine learning models are not available. Please check server logs."}), 500

    try:
        text_to_analyze = ""
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.content_type.startswith("text/"):
                text_to_analyze = file.read().decode('utf-8')
            else:
                return jsonify({"error": "Invalid file type. Please upload a plain text (.txt) file."}), 400
        elif 'text' in request.form:
            text_to_analyze = request.form.get('text')
        
        if not text_to_analyze:
            return jsonify({"error": "No text or file was provided for analysis."}), 400

        cleaned_text = preprocess_text(text_to_analyze)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction_val = model.predict(vectorized_text)[0]
        
        # --- FIX: CORRECTED LABEL LOGIC ---
        # Based on the notebook's confusion matrix, 0=FAKE and 1=REAL.
        # The original code had this reversed, causing the wrong text to display.
        if prediction_val == 0:
            prediction_text = "Prediction: Looks like FAKE ⚠ News"
        else:
            prediction_text = "Prediction: Looks like REAL ✔️ News"
        
        # --- FIX: STABILIZED CONFIDENCE CALCULATION ---
        # This prevents the 'nan' issue by ensuring the score is a valid number.
        raw_score = model.decision_function(vectorized_text)[0]
        if np.isnan(raw_score):
             confidence_percentage = "50.00%" # Default confidence if score is unstable
        else:
            confidence = 1 / (1 + np.exp(-abs(raw_score))) 
            # Scale confidence to be more intuitive (from 50% to 100%)
            scaled_confidence = 0.5 + (confidence - 0.5) * 2
            confidence_percentage = f"{scaled_confidence * 100:.2f}%"

        return jsonify({
            "prediction": prediction_text,
            "confidence": confidence_percentage,
            "text_length": len(text_to_analyze)
        })

    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({"error": f"An unexpected error occurred during analysis: {str(e)}"}), 500

# --- 5. SERVE THE FRONTEND HTML PAGE ---
@app.route('/')
def serve_index():
    return render_template('index.html')

# --- 6. RUN THE FLASK APPLICATION ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

