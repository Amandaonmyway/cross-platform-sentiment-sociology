import pandas as pd
from src.sentiment_analyzer import SentimentAnalyzer

def main():
    # Simulate raw text samples collected from cross-platform sources
    sample_data = pd.DataFrame({
        'platform': ['Reddit', 'NicheForum', 'Reddit'],
        'raw_text': [
            "Check out this cool research: https://example.com #Sociology",
            "@user Great discussion on digital polarization today!",
            "Extreme political division is damaging public discourse..."
        ]
    })
    
    print("--- Raw Text Data ---")
    print(sample_data)
    
    # Initialize the SentimentAnalyzer and execute the pipeline
    analyzer = SentimentAnalyzer()
    processed_df = analyzer.process_dataframe(sample_data, text_column='raw_text')
    
    print("\n--- Processed Sentiment & Polarization Results ---")
    print(processed_df[['platform', 'raw_text', 'compound', 'sentiment_label']])

if __name__ == "__main__":
    main()
