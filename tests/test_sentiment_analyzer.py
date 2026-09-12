import pytest
import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer

def test_clean_text():
    analyzer = SentimentAnalyzer()
    raw = "Check out https://example.com #Sociology @user!"
    cleaned = analyzer.clean_text(raw)
    assert "http" not in cleaned
    assert "#" not in cleaned
    assert "@" not in cleaned

def test_compute_sentiment():
    analyzer = SentimentAnalyzer()
    res_pos = analyzer.compute_sentiment("This is brilliant and amazing!")
    assert res_pos['compound'] >= 0.05
    
    res_neg = analyzer.compute_sentiment("Absolute disaster, worst ever!")
    assert res_neg['compound'] <= -0.05

def test_process_dataframe():
    analyzer = SentimentAnalyzer()
    df = pd.DataFrame({
        'platform': ['Reddit'],
        'text': ["Completely wonderful development!"]
    })
    processed = analyzer.process_dataframe(df, 'text')
    assert 'sentiment_label' in processed.columns
    assert processed.loc[0, 'sentiment_label'] == 'Positive'
