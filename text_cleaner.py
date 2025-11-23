import re
import nltk
from nltk.corpus import stopwords

# TEXT CLEANER FUNCTION
def clean_text(text):
    text = text.lower()   #  Lowercase conversion
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # URL removal
    text = re.sub(r'<.*?>', '', text) # HTML tag removal
    text = re.sub(r'[^a-z\s]', '', text) # Special character removal
    text = re.sub(r'\s+', ' ', text).strip() # Extra whitespace removal   #strip to remove forward or backward spcaes
    text = re.sub(r'\S+@\S+\.\S+', '', text) # email removal
    text = re.sub(r'[^\w\s]', '', text) # punctuations
    text = re.sub(r'(.)\1{2,}', r'\1', text) # Repeated Characters / Elongated Words
    text = re.sub(r'#', '', text) # Hashtags (like from Twitter/Instagram)
    text= re.sub(r'@\w+', '', text) # Mention username
    text= re.sub(r'[^\x00-\x7F]+', '', text) # non ASCII characters (emojis , foreign characcters)
    return text


# STOP WORDS, TOKENISATION AND LEMMATIZATION
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
#creating functions out of it.
def tokenize_and_lemmatize(text):
    """
    This function breaks down the text into words, removes unnecessary words (stop words),
    and turns words into their base forms (lemmas).
    """
    try:
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Filter out stopwords and punctuation, then lemmatize
        filtered_words = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower not in stop_words and token_lower not in punctuation:
                lemma = lemmatizer.lemmatize(token_lower)
                filtered_words.append(lemma)

        return filtered_words

    except Exception as e:
        return f"Error: {e}"

