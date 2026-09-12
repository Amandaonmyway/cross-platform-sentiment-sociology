import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    """Cross-platform text sentiment and polarization analyzer."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text: str) -> str:
        """Clean raw social media text by removing URLs, mentions, and special characters."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+|\#", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]", "", text)
        return text.strip().lower()

    def compute_sentiment(self, text: str) -> dict:
        """Compute sentiment scores for a single text string."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        return self.sia.polarity_scores(cleaned)

    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process a dataframe in batch and return a new dataframe with sentiment polarization metrics."""
        sentiments = df[text_column].apply(self.compute_sentiment)
        sentiment_df = pd.DataFrame(list(sentiments))
        
        # Merge results
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        # Assign sentiment polarization labels (Positive, Neutral, Negative)
        result_df['sentiment_label'] = result_df['compound'].apply(
            lambda c: 'Positive' if c >= 0.05 else ('Negative' if c <= -0.05 else 'Neutral')
        )
        return result_df
