# Fake News Detection System 📰

A machine learning-based web application designed to classify news articles as either "Real" or "Fake" using Natural Language Processing (NLP) techniques. This project demonstrates a complete end-to-end data science and machine learning pipeline with a user-friendly Flask web interface.

## 🎯 Project Overview

This project showcases a practical implementation of machine learning for combating misinformation. The system analyzes text content using advanced NLP techniques and predicts the authenticity of news articles with high accuracy. The entire workflow, from data preprocessing and model training to web-based deployment, is handled within this repository.

## ✨ Key Features

### Core Functionality
- **Interactive Web Interface**: User-friendly Flask application for instant news verification
- **High-Performance ML Model**: Passive Aggressive Classifier achieving **95.57% accuracy**
- **Advanced NLP Processing**: Text preprocessing with tokenization, lemmatization, and stop-word removal
- **Real-time Predictions**: Instant classification of user-inputted news text
- **Responsive Design**: Modern web interface built with Bootstrap and Tailwind CSS

### Technical Highlights
- **TF-IDF Vectorization**: Converts text into numerical features for ML processing
- **Online Learning Algorithm**: Efficient Passive Aggressive Classifier for large-scale text classification
- **Robust Text Processing**: Handles various text formats and cleaning requirements
- **Model Persistence**: Trained model and vectorizer saved as pickle files for deployment

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **Scikit-learn**: Model training and evaluation
- **NLTK**: Natural Language Processing and text preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization

### Web Development
- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, Bootstrap 5, Tailwind CSS
- **Templates**: Jinja2 templating engine

### Development Tools
- **Jupyter Notebook**: Model development and experimentation
- **Pickle**: Model serialization and deployment

## 🧠 How It Works

The system follows a comprehensive machine learning pipeline:

### 1. Data Preprocessing
- Raw news dataset (`train.csv`) is loaded and cleaned
- Missing values are handled and unnecessary columns removed
- Text data is normalized and prepared for feature extraction

### 2. Feature Extraction
- **TF-IDF Vectorization**: Transforms text into numerical features
- Captures the importance of words relative to the document and corpus
- Creates a sparse matrix suitable for machine learning algorithms

### 3. Model Training
- **Passive Aggressive Classifier**: An online learning algorithm ideal for text classification
- Efficient for large-scale datasets with good performance on text data
- Trained on vectorized news articles with binary classification (Real/Fake)

### 4. Text Processing Pipeline
```python
# Text preprocessing steps:
1. Remove special characters and punctuation
2. Convert to lowercase
3. Tokenization using NLTK
4. Remove stopwords
5. Lemmatization for word normalization
6. TF-IDF vectorization
7. Model prediction
```

### 5. Web Application
- Flask backend serves the ML model
- Real-time text processing and prediction
- User-friendly interface for news input and result display

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HarpreetSingh3500/fake-news-detection.git
   cd fake-news-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (handled automatically by the app)
   ```python
   # The app automatically downloads required NLTK data:
   # - stopwords
   # - punkt tokenizer
   # - wordnet lemmatizer
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your web browser and navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

```
fake-news-detection/
├── app.py                          # Main Flask application
├── Fake_News_Detector-PA.ipynb     # Jupyter notebook with ML pipeline
├── requirements.txt                # Project dependencies
├── model.pkl                       # Trained ML model (serialized)
├── vector.pkl                      # Trained TF-IDF vectorizer (serialized)
├── README.md                       # Project documentation
├── templates/                      # HTML templates
│   ├── index.html                  # Homepage
│   ├── about.html                  # About page
│   └── prediction.html             # Prediction interface
└── static/                         # Static assets
    ├── hero_img.svg                # Hero section illustration
    └── icons/                      # Favicon and icons
```

## 💻 Usage

### Web Interface Navigation

1. **Home Page** (`/`)
   - Welcome screen with project overview
   - Clean, modern design with call-to-action
   - Navigation to prediction and about sections

2. **Prediction Page** (`/predict`)
   - Text input form for news articles
   - Real-time prediction results
   - User-friendly feedback with emojis

3. **About Page** (`/about`)
   - Project information and technologies used
   - Developer contact and GitHub repository link

### Making Predictions

1. Navigate to the **Prediction** page
2. Enter the news headline or article text in the input field
3. Click the **"Predict"** button
4. View the result:
   - 📰 **"Looking Real News"** - Article appears authentic
   - 📰 **"Looking Fake News"** - Article appears fabricated

### Example Usage
```
Input: "Scientists discover new planet in solar system"
Output: "Prediction of the News: Looking Fake News📰"

Input: "Government announces new economic policy measures"
Output: "Prediction of the News: Looking Real News📰"
```

## 🔧 Model Performance

### Accuracy Metrics
- **Overall Accuracy**: 95.57%
- **Model Type**: Passive Aggressive Classifier
- **Feature Extraction**: TF-IDF Vectorization
- **Text Processing**: NLTK-based preprocessing

### Model Characteristics
- **Fast Prediction**: Optimized for real-time classification
- **Memory Efficient**: Suitable for web deployment
- **Robust Processing**: Handles various text formats and lengths
- **High Precision**: Reliable fake news detection

## 🔍 Technical Implementation

### Flask Application Structure
```python
# Key components:
- NLTK data management with SSL handling
- Model loading and caching
- Text preprocessing pipeline
- Real-time prediction endpoint
- Responsive web interface
```

### Text Processing Pipeline
```python
def fake_news_det(news):
    # 1. Clean text (remove special characters)
    # 2. Convert to lowercase
    # 3. Tokenize using NLTK
    # 4. Remove stopwords
    # 5. Apply lemmatization
    # 6. Vectorize using trained TF-IDF
    # 7. Predict using trained model
    return prediction
```

## 🌟 Future Enhancements

### Advanced Models
- **Deep Learning**: Implement LSTM or Transformer-based models (BERT, RoBERTa)
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Fine-tuning**: Use pre-trained language models for better performance

### Features & Integration
- **Real-time Data**: Integration with NewsAPI for live news verification
- **Batch Processing**: Upload and process multiple articles simultaneously
- **Confidence Scores**: Provide prediction confidence percentages
- **Source Analysis**: Include source credibility assessment

### Deployment & Scalability
- **Docker Containerization**: Easy deployment across different environments
- **Cloud Deployment**: AWS, Heroku, or Google Cloud integration
- **API Development**: RESTful API for third-party integrations
- **Database Integration**: Store predictions and user feedback

### User Experience
- **Browser Extension**: Chrome/Firefox extension for real-time news checking
- **Mobile App**: React Native or Flutter mobile application
- **Advanced Analytics**: Dashboard with prediction statistics and trends

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines for Python code
- Add comments and docstrings for new functions
- Update documentation for new features
- Include tests for new functionality

## 📊 Dataset Information

The model is trained on a comprehensive news dataset containing:
- **Real News**: Authentic articles from verified news sources
- **Fake News**: Fabricated or misleading articles
- **Features**: Article text, headlines, and metadata
- **Preprocessing**: Cleaned and tokenized text data

## 🛡️ Limitations & Considerations

### Current Limitations
- **Language Support**: Currently optimized for English text
- **Context Sensitivity**: May struggle with satirical or opinion pieces
- **Evolving Misinformation**: New fake news patterns may require model retraining
- **Bias Considerations**: Model performance may vary across different news topics

### Best Practices
- **Human Verification**: Use as a screening tool, not definitive judgment
- **Regular Updates**: Retrain model with new data periodically
- **Context Awareness**: Consider article source and publication date
- **Multiple Sources**: Cross-reference with other fact-checking tools

## 👨‍💻 Author

**Harpreet Singh** - *Project Developer*
- GitHub: [@HarpreetSingh3500](https://github.com/HarpreetSingh3500)
- Project Link: [Fake News Detection](https://github.com/HarpreetSingh3500/fake-news-detection)

## 🙏 Acknowledgments

- **NLTK Team** for natural language processing tools
- **Scikit-learn** for machine learning algorithms
- **Flask Community** for web framework
- **Bootstrap & Tailwind** for responsive design components
- **Open Source Community** for inspiration and resources

---

**Fight Misinformation with Machine Learning! 🤖📰**
