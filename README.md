# Text_Analysis_final_for_deployment

Spacy model need to be insatted , command to do so :- python -m spacy download en_core_web_sm
NLTK requires a few resources (punkt, stopwords, wordnet)
commands to do so :- 

import nltk

# Run this file once to download the necessary data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
print("Download complete!")
