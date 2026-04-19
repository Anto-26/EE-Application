from src.dataloading.loader import DataLoader
from src.nlp.preprocess import TextPreprocessor
from src.nlp.sentiment import SentimentAnalyzer


def main():

    # 1. LOAD DATA
    path = "/Users/jamesantoarnoldj/Desktop/Projects/EE App/data/02-preprocessed/ee_data_clean.csv"

    loader = DataLoader(path)
    df = loader.load()

    print("\nDATA LOADED")
    print(df.head())
    print(df.columns)

    # 2. PREPROCESS TEXT
    preprocessor = TextPreprocessor()
    df = preprocessor.process_dataframe(df)

    print("\nTEXT PREPROCESSED")
    print(df[["likes_clean", "dislikes_clean", "combined_text"]].head())

    # 3. SENTIMENT ANALYSIS
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_row_sentiment(df)

    print("\nSENTIMENT ADDED")
    print(df[
        ["likes_sentiment", "dislikes_sentiment", "overall_sentiment"]
    ].head())

    # 4. DEPARTMENT INSIGHTS
    dept_sentiment = analyzer.department_sentiment(df)

    print("\nDEPARTMENT SENTIMENT")
    print(dept_sentiment)

    # 5. FINAL CHECKS
    print("\nSHAPE:", df.shape)

    print("\nNULL CHECK:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()