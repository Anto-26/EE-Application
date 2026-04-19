import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_score(self, text: str) -> float:
        """
        Returns compound sentiment score (-1 to +1)
        """
        if not isinstance(text, str) or text.strip() == "":
            return 0.0

        return self.analyzer.polarity_scores(text)["compound"]

    def analyze_row_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds sentiment scores to each row
        """

        # Likes sentiment
        df["likes_sentiment"] = df["likes_clean"].apply(self.get_sentiment_score)

        # Dislikes sentiment
        df["dislikes_sentiment"] = df["dislikes_clean"].apply(self.get_sentiment_score)

        # Combined sentiment (overall employee mood)
        df["overall_sentiment"] = (
            df["likes_sentiment"] + df["dislikes_sentiment"]
        ) / 2

        return df

    def department_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregated sentiment by department
        """

        dept_sentiment = df.groupby("Department").agg(
            avg_likes_sentiment=("likes_sentiment", "mean"),
            avg_dislikes_sentiment=("dislikes_sentiment", "mean"),
            avg_overall_sentiment=("overall_sentiment", "mean"),
            avg_rating=("Overall_rating", "mean"),
            count=("Department", "count")
        ).reset_index()

        return dept_sentiment.sort_values("avg_overall_sentiment")

    def location_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregated sentiment by location
        """

        loc_sentiment = df.groupby("Place").agg(
            avg_likes_sentiment=("likes_sentiment", "mean"),
            avg_dislikes_sentiment=("dislikes_sentiment", "mean"),
            avg_overall_sentiment=("overall_sentiment", "mean"),
            avg_rating=("Overall_rating", "mean"),
            count=("Place", "count")
        ).reset_index()

        return loc_sentiment.sort_values("avg_overall_sentiment")

    def jobtype_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sentiment by job type
        """

        job_sentiment = df.groupby("Job_type").agg(
            avg_overall_sentiment=("overall_sentiment", "mean"),
            avg_rating=("Overall_rating", "mean"),
            count=("Job_type", "count")
        ).reset_index()

        return job_sentiment

    def run_full_analysis(self, df: pd.DataFrame) -> dict:
        """
        Runs complete sentiment pipeline
        """

        df = self.analyze_row_sentiment(df)

        results = {
            "data": df,
            "department_sentiment": self.department_sentiment(df),
            "location_sentiment": self.location_sentiment(df),
            "jobtype_sentiment": self.jobtype_sentiment(df),
        }

        return results


# ----------------------------------
# Quick test run
# ----------------------------------
if __name__ == "__main__":
    # Example test (replace with real data from preprocess step)
    sample = {
        "likes_clean": [
            "great team flexible hours supportive culture",
            "good management learning opportunities"
        ],
        "dislikes_clean": [
            "low salary poor management no growth",
            "high workload stressful environment"
        ],
        "Department": ["IT", "Sales"],
        "Place": ["London", "Manchester"],
        "Job_type": ["Full Time", "Full Time"],
        "Overall_rating": [3.2, 2.8]
    }

    df = pd.DataFrame(sample)

    analyzer = SentimentAnalyzer()
    results = analyzer.run_full_analysis(df)

    print("\n📊 Department Sentiment")
    print(results["department_sentiment"])

    print("\n📍 Location Sentiment")
    print(results["location_sentiment"])

    print("\n🧑‍💼 Job Type Sentiment")
    print(results["jobtype_sentiment"])

    print("\n✅ Row-level data sample")
    print(results["data"].head())