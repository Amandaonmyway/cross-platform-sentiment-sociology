import pandas as pd
import re

def clean_text(text):
    """Clean raw social media text by removing URLs, mentions, and special characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.strip().lower()

def calculate_basic_sentiment(df, text_column):
    """Placeholder function for cross-platform sentiment polarity scoring."""
    df['cleaned_text'] = df[text_column].apply(clean_text)
    return df
