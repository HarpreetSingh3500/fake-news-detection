import nltk
try:
        # Download 'stopwords' and 'punkt' to a accessible location
        nltk.download('stopwords', download_dir='/opt/render/project/src/.nltk_data', quiet=True)
        nltk.download('punkt', download_dir='/opt/render/project/src/.nltk_data', quiet=True)
        print("NLTK corpora downloaded successfully.")
except Exception as e:
        print(f"Error downloading NLTK corpora: {e}")
    
