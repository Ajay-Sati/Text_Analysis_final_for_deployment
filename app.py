import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from text_cleaner import clean_text, tokenize_and_lemmatize
from nlp_functions import (
    show_wordcloud,plot_top_ngrams_bar_chart,detect_emotions,detect_overall_sentiment_avg,classify_custom, safe_summarize
)

st.title("INTERACTIVE TEXT ANALYSIS PLATFORM.")
st.divider()
# Just add it after st.sidebar:
a = st.sidebar.radio("SELECT ONE: ", ["Process Textual Data", "Process Csv file"])

if a=="Process Textual Data":
    st.header("Input your textual data")
    text = st.text_area("📝 Enter your text below:", height=150)

    if st.button("🔍 Analyze"):
        if not text.strip():
            st.warning("⚠️ Please enter some text for analysis.")
        else:
            st.success("✅ Analysis complete!")

            # 1. Clean & Preprocess
            cleaned = clean_text(text)
            tokens = tokenize_and_lemmatize(cleaned)

            st.subheader("## ✨ Cleaned & Lemmatized Text")
            st.write(" ".join(tokens) if tokens else "No meaningful tokens extracted.")
            st.divider()

            # 2. Word Cloud
            if tokens:
                st.markdown("## ☁️ Word Cloud")
                wc_plot = show_wordcloud(tokens)
                st.pyplot(wc_plot)

            st.divider()
            number = st.number_input("Enter n  value for count of words in grams", min_value=2, step=1)
            # 3. N_GRAM ANALYSIS
            st.subheader("## N-GRAM ANALYSIS")
            plot_top_ngrams_bar_chart(tokens, gram_n=number)
            st.divider()

            # 4. TONE [EMOTION] DETECTION
            st.subheader("## EMOTION DETECTION")
            top_emotions_df = detect_emotions(text)
            max_index = top_emotions_df['Score'].idxmax()
            Emotion = top_emotions_df.loc[max_index, 'Emotion']
            score = top_emotions_df.loc[max_index, 'Score']
            st.write(f"Preidcted Emotion(Tone):- {Emotion}, with {score}% confidence")

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            with col1:
                # Display table of top 5 emotions
                st.subheader("Top 5 Emotions")
                st.dataframe(top_emotions_df)

            with col2:
                # Display Plotly chart
                st.subheader("Confidence Bar Chart")
                fig = px.bar(top_emotions_df, x="Emotion", y="Score", color="Emotion"
                             , text_auto='.4f')
                fig.update_layout( template='plotly_white')
                st.plotly_chart(fig)

            st.divider()

            # 5 SENTIMENT ANALYSIS
            st.subheader("## SENTIMENT  DETECTION")
            result = detect_overall_sentiment_avg(text)
            if "error" in result:
                st.write("Error:", result["error"])
            else:
                st.write("Overall Sentiment:", result["overall_sentiment"])
                st.write("Average Scores:",
                         pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))

            # SENTENCE TYPE CLASSIFICATION.
            st.markdown("## SENTENCE TYPE CLASSIFICATION")
            output = classify_custom(text)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 🔍 Predicted: **{output['predicted_category']} (Score: {output['score']:.2f})**")
                st.write("📊 Top Categories:")
                for label, score in output["all_categories"][:5]:  # Show top 3
                    st.write(f"  - {label}: {score:.2f}")

            with col2:
                labels = []
                scores = []
                for label, score in output["all_categories"][:5]:
                    labels.append(label)
                    scores.append(score)

                fig = px.bar(
                    x=labels,
                    y=scores,
                    color=labels,
                    title="Top 5 sentence type classification.",
                    labels={"Value": "Value Count"},
                    height=400
                )

                st.plotly_chart(fig)

            # SUMMARY GENERATION
            st.markdown("## SUMMARY GENERATION")
            output = safe_summarize(text)
            st.write(output)

if a=="Process Csv file":
    st.header("Upload your CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully uploaded!")

        st.divider()
        st.subheader("Step 2: Choose filtering options")

        # Step 3: Let user choose a column to filter
        column_name = st.selectbox("Select a column to filter the data", df.columns)

        # Step 4: Show unique values from that column
        unique_vals = df[column_name].dropna().unique()

        # Step 5: Let user pick values to filter
        selected_value = st.multiselect(f"Choose value(s) from '{column_name}' to filter", unique_vals)

        # Step 6: Select the textual column for analysis
        text_processing_column = st.selectbox("Select the column for text analysis", df.columns)

        # Step 7: Filtered view
        if selected_value:
            filtered_df = df[df[column_name].isin(selected_value)].sample(500)
            filtered_df = filtered_df[text_processing_column]
            st.subheader("Filtered Text Data")
            st.dataframe(filtered_df)

            st.divider()
            st.subheader("Step 3: NLP Analysis")

            # Combine all filtered text
            text = " ".join(filtered_df.dropna().astype(str))

            # 1. Clean & Preprocess
            cleaned = clean_text(text)
            tokens = tokenize_and_lemmatize(cleaned)

            st.markdown("## ✨ Cleaned & Lemmatized Text")
            st.write(" ".join(tokens) if tokens else "No meaningful tokens extracted.")
            st.divider()

            # 2. Word Cloud
            if tokens:
                st.markdown("## ☁️ Word Cloud")
                wc_plot = show_wordcloud(tokens)
                st.pyplot(wc_plot)

            st.divider()

            # 3. N_GRAM ANALYSIS
            st.markdown("## N-GRAM ANALYSIS")
            plot_top_ngrams_bar_chart(tokens, gram_n=3)
            st.divider()

            # 4. TONE [EMOTION] DETECTION
            st.markdown("## EMOTION DETECTION")
            top_emotions_df = detect_emotions(text)
            max_index = top_emotions_df['Score'].idxmax()
            Emotion = top_emotions_df.loc[max_index, 'Emotion']
            score = top_emotions_df.loc[max_index, 'Score']
            st.write(f"Preidcted Emotion(Tone):- {Emotion}, with {score}  confidence")

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)

            with col1:
                # Display table of top 5 emotions
                st.subheader("Top 5 Emotions")
                st.dataframe(top_emotions_df)

            with col2:
                # Display Plotly chart
                st.subheader("Confidence Bar Chart")
                fig = px.bar(top_emotions_df, x="Emotion", y="Score", color="Emotion"
                             , text_auto='.4f')
                fig.update_layout( template='plotly_white')
                st.plotly_chart(fig)

            st.divider()

            # 5 SENTIMENT ANALYSIS
            st.markdown("## SENTIMENT  DETECTION")
            result = detect_overall_sentiment_avg(text)
            if "error" in result:
                st.write("Error:", result["error"])
            else:
                st.write("Overall Sentiment:", result["overall_sentiment"])
                st.write("Average Scores:",
                         pd.DataFrame(list(result['average_scores'].items()), columns=['Emotion', 'Score']))

            # SENTENCE TYPE CLASSIFICATION.
            st.markdown("#TONE OF SENTENCE OR TEXT")
            output = classify_custom(text)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 🔍 Predicted: **{output['predicted_category']} (Score: {output['score']:.2f})**")
                st.write("📊 Top Categories:")
                for label, score in output["all_categories"][:5]:  # Show top 3
                    st.write(" ")
                    st.write(f"  - {label}: {score:.2f}")

            with col2:
                labels = []
                scores = []
                for label, score in output["all_categories"][:5]:
                    labels.append(label)
                    scores.append(score)

                fig = px.bar(
                    x=labels,
                    y=scores,
                    color=labels,
                    title="Top 5 sentence type classification.",
                    labels={"Value": "Value Count"},
                    height=400
                )

                st.plotly_chart(fig)

            # SUMMARY GENERATION
            st.markdown("## SUMMARY GENERATION")
            output = safe_summarize(text)
            st.write(output)


