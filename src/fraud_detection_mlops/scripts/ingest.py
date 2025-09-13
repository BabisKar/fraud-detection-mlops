import pandas as pd
import argparse
import pathlib
from sklearn.model_selection import train_test_split
import logging
from src.fraud_detection_mlops.utils.utils import load_config
logger = logging.getLogger(__name__)
logging.basicConfig()

def main(logger: logging.Logger,raw_path:str, processed_dir: str, test_size: float, random_state: int):

    logger.warning(f"Starting data ingestion from {raw_path}")
    try:
        raw_path = pathlib.Path(raw_path)
        processed_dir = pathlib.Path(processed_dir)
    except Exception as e:
        logger.error(f"Error with file paths: {e}")
        raise
    # Load the raw data
    df = pd.read_csv(raw_path)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['Class'])

    # Ensure the processed directory exists
    processed_path = pathlib.Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Save the training and testing sets to CSV files
    train_df.to_csv(processed_path / 'train.csv', index=False)
    test_df.to_csv(processed_path / 'test.csv', index=False)

    print(
        f"Data ingested successfully. Training data saved to {processed_path / 'train.csv'} and testing data saved to {processed_path / 'test.csv'}.")


if __name__ == "__main__":
    loaded_config = load_config(logger=logger,yaml_file_name='ingest.yaml')
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default=loaded_config['raw_path'] , help='Path to the raw CSV file')
    parser.add_argument('--processed_dir', type=str, default=loaded_config['processed_dir'], help='Directory to save processed CSV files')
    parser.add_argument('--test_size', type=float, default=loaded_config['test_size'],
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--random_state', type=int, default=loaded_config['random_state'], help='Random state for reproducibility')
    args = parser.parse_args()

    main(logger,args.raw_path, args.processed_dir, args.test_size, args.random_state)
