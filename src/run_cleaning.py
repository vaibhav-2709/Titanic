from data_loader import load_csv
from data_cleaning import clean_data


RAW_DATA_PATH = "../data/raw/Titanic-Dataset.csv"
CLEAN_DATA_PATH = "../data/processed/cleaned_data.csv"


def main():
    print("Loading raw dataset...")
    df = load_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")
    cleaned_df = clean_data(df)

    print("Saving cleaned dataset...")
    cleaned_df.to_csv(CLEAN_DATA_PATH, index=False)

    print("Data cleaning completed successfully.")


if __name__ == "__main__":
    main()
