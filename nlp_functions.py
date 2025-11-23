# word clpud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# n-gram analysis
from nltk.util import ngrams
from collections import Counter
import plotly.graph_objects as go
import streamlit as st


#CREATING CHUNKS
import spacy

# EMOTIONAL ANALYSIS
from transformers import pipeline
import pandas as pd
import plotly.express as px


#

# WORD CLOUD.
def show_wordcloud(tokens):
    """
    Generate a WordCloud from a list of tokens.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        return plt
    except Exception as e:
        return f"Error generating word cloud: {e}"


# N-GRAM ANLYSIS
def plot_top_ngrams_bar_chart(tokens, gram_n=2, top_n=15):
    try:
        # Step 1: Create n-grams
        ngram_list = list(ngrams(tokens, gram_n))

        # Step 2: Count most common n-grams
        ngram_counts = Counter(ngram_list).most_common(top_n)

        if not ngram_counts:
            raise ValueError("No n-grams found in the given token list.")

        # Step 3: Prepare labels and counts
        labels = []
        counts = []
        for ngram, count in ngram_counts:
            labels.append(' '.join(ngram))
            counts.append(count)

        # Step 5: Plotly Bar Chart
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=counts,
                text=counts,
                textposition='outside'
            )
        ])

        # Step 6: Layout settings
        fig.update_layout(
            title=f"Top {top_n} {gram_n}-grams",
            xaxis_title=f"{gram_n}-grams",
            yaxis_title="Frequency",
            xaxis_tickangle=-45,
            template='plotly_white',
            margin=dict(t=50, b=120),
            height=500
        )

        st.plotly_chart(fig)

    except Exception as e:
        print(f"An error occurred: {e}")


#SPLITTING INTO CHUNKS
# Load the English model
nlp = spacy.load("en_core_web_sm")

# Function to split long text into sentence-based chunks
def split_text_into_chunks_spacy(text, max_length=500):
    doc = nlp(text)
    chunks = []
    current_chunk = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence   # start a new chunk with the current sentence. [  sentence that was not the part of last chunk]
    if current_chunk: # Is there any text left in current_chunk?At the end of the loop, there might still be some text in current_chunk that was not yet saved to the chunks list. We don’t want to lose that.
        chunks.append(current_chunk.strip())
    return chunks



# EMOTION ANALYSIS
model_name = "nateraw/bert-base-uncased-emotion"
emotion_model = pipeline("text-classification", model=model_name, top_k=None)

def detect_emotions(text):
    chunks = split_text_into_chunks_spacy(text)
    emotion_totals = {}
    emotion_counts = {}

    for chunk in chunks:
        results = emotion_model(chunk)[0]
        for result in results:
            label = result['label']
            score = result['score']
            emotion_totals[label] = emotion_totals.get(label, 0) + score
            emotion_counts[label] = emotion_counts.get(label, 0) + 1

    # Compute average scores
    emotion_averages = {label: emotion_totals[label] / emotion_counts[label] for label in emotion_totals}

    # Sort and return top 5 emotions
    sorted_emotions = sorted(emotion_averages.items(), key=lambda x: x[1], reverse=True)
    top_5 = sorted_emotions[:5]

    # Create and return DataFrame
    df = pd.DataFrame(top_5, columns=["Emotion", "Score"])
    return df



# SENTIMENTAL ANALYSIS
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, return_all_scores=True)


# Function to get overall sentiment based on average scores
def detect_overall_sentiment_avg(text):
    try:
        sentiment_labels = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive'
        }

        chunks = split_text_into_chunks_spacy(text)
        score_totals = {'Negative': 0.0, 'Neutral': 0.0, 'Positive': 0.0}
        chunk_count = len(chunks)

        for chunk in chunks:
            results = sentiment_classifier(chunk)[0]  # return_all_scores=True gives list of dicts
            for res in results:
                label = sentiment_labels[res['label']]
                score_totals[label] += res['score']

        avg_scores = {}

        for label in score_totals:
            avg_scores[label] = score_totals[label] / chunk_count

        # Select sentiment with highest average score
        overall_sentiment = max(avg_scores, key=avg_scores.get)

        return {
            "overall_sentiment": overall_sentiment,
            "average_scores": avg_scores,
            "total_chunks": chunk_count
        }

    except Exception as e:
        return {"error": str(e)}


# TONE OF SENTENCE TEXT
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = [
    "factual",
    "opinion",
    "question",
    "command",
    "emotion",
    "personal experience",
    "suggestion",
    "story",
    "prediction",
    "warning",
    "instruction",
    "definition",
    "narrative",
    "news",
    "argument"
]

def classify_custom(text):
    result = classifier(text, candidate_labels=labels)
    return {
        "text": text,
        "predicted_category": result["labels"][0],
        "score": result["scores"][0],
        "all_categories": list(zip(result["labels"], result["scores"]))
    }


# SUMMARY GENERATION
# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# function for creating summary of each chunks and then creating summary out of the chunks of summary.
from transformers import pipeline
import math

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# function for creating summary of each chunks
def safe_summarize(text, max_length=300, min_length=100, chunk_size=500):
    """
    A helper to safely summarize text under token limits.
    """
    if len(text.split()) <= chunk_size:
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    else:
        chunks = split_text_into_chunks_spacy(text, chunk_size)
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"⤷ Final summary chunk {i+1} of {len(chunks)}...")
            result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(result[0]['summary_text'])
        return safe_summarize(" ".join(summaries), max_length, min_length, chunk_size) # recurrsive function


    
