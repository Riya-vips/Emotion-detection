import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, source_path):
        self.source_path = source_path

    def initiate_data_ingestion(self):
        # Load dataset
        df = pd.read_csv(self.source_path)

        # Clean columns and data
        df.columns = df.columns.str.lower().str.strip()
        df.drop_duplicates(inplace=True)
        df.dropna(subset=['text', 'emotion'], inplace=True)

        # Stratified train-test split
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['emotion']
        )
        return train_set, test_set, df


if __name__ == '__main__':
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    source_data_path = 'notebook/data/emotion_dataset_raw.csv'

    ingestion = DataIngestion(source_data_path)
    train_df, test_df, raw_df = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _, _ = transformation.initiate_data_transformation(raw_df)

    trainer = ModelTrainer()
    report = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)

    print("Model evaluation report:")
    print(report)
