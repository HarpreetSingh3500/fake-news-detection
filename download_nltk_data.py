import nltk

try:
    # Attempt to download the 'stopwords' corpus
    nltk.download('stopwords', download_dir='/opt/render/project/src/.nltk_data', quiet=True)
    # Attempt to download the 'punkt' corpus, which is also needed for tokenization
    nltk.download('punkt', download_dir='/opt/render/project/src/.nltk_data', quiet=True)
    print("NLTK corpora downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK corpora: {e}")
