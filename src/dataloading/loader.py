from pathlib import Path
import pandas as pd

# Expected dataset columns
EXPECTED_COLUMNS = [
    "Title",
    "Place",
    "Job_type",
    "Department",
    "Date",
    "Overall_rating",
    "work_life_balance",
    "skill_development",
    "salary_and_benefits",
    "job_security",
    "career_growth",
    "work_satisfaction",
    "Likes",
    "Dislikes",
]


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
    
    def file_exists(self) -> bool:
        return self.filepath.exists()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV file.
        """
        if not self.file_exists():
            raise FileNotFoundError(f"File Not Found:{self.filepath}")
        
        suffix = self.filepath.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(self.filepath)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(self.filepath)
        else:
            raise ValueError("Unsupported file type. Please use CSV or Excel File.")
        return df
        
    def strandardise_columns(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Strip spaces from the column names.
        """
        df.columns = df.columns.str.strip()
        return df
    
    def validate_columns(self, df:pd.DataFrame):
        """
        Check if the required columns are present in the dataset.
        """
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]

        if missing:
            raise ValueError(f"Missing Columns: {missing}")
    
    def load(self) -> pd.DataFrame:
        """
        Full Pipeline.
        """
        df = self.load_data()
        df = self.strandardise_columns(df)
        self.validate_columns(df)
        
        return df
    

# ----------------------------------
# Quick test run
# ----------------------------------
if __name__ == "__main__":
    path = "/Users/jamesantoarnoldj/Desktop/Projects/EE App/data/02-preprocessed/ee_data_clean.csv"

    loader = DataLoader(path)
    df = loader.load()

    print("Dataset Loaded Successfully")
    print(df.head())
    print(df.info())