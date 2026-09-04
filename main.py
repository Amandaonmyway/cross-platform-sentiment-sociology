import pandas as pd
from src.sentiment_analyzer import calculate_basic_sentiment

def main():
    # Simulate raw text samples collected from cross-platform sources
    sample_data = pd.DataFrame({
        'raw_text': [
            "Check out this cool research: https://example.com #Sociology",
            "@user Great discussion on digital polarization today!",
            "Extreme political division is damaging public discourse..."
        ]
    })
    
    print("--- Raw Text Data ---")
    print(sample_data)
    
    # Call the core module from the src directory for cleaning and preprocessing
    processed_df = calculate_basic_sentiment(sample_data, 'raw_text')
    
    print("\n--- Cleaned and Processed Computational Sociology Text ---")
    print(processed_df[['cleaned_text']])

if __name__ == "__main__":
    main()
