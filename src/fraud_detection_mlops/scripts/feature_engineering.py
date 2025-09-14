import pandas as pd
import logging
from src.fraud_detection_mlops.utils.utils import load_config
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pathlib
import os
import sys

def create_features(logger: logging.Logger, input_data_path : str ,output_data_path: str, time_bins : list,
                    time_labels: list, amount_labels: list , amount_bins : list, dataset: str,
                    polynomial_degree: int ) -> pd.DataFrame:
    """
    Create new features for the fraud detection dataset.
    Args:
        logger (logging.Logger): Logger object for logging messages.
        data_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Dataframe with new features added.
    """

    logger.info('Starting feature engineering process')

    # Load the dataset

    try:
        input_data_path = pathlib.Path(input_data_path)
        output_data_path = pathlib.Path(output_data_path)
    except Exception as e:
        logger.error(f"Error with file paths: {e}")
        raise

    if dataset == 'train':
        logger.info('Processing training dataset')
        data = pd.read_csv(os.path.join(input_data_path, 'train.csv'))
    elif dataset == 'test':
        logger.info('Processing testing dataset')
        data = pd.read_csv(os.path.join(input_data_path, 'test.csv'))
    else:
        logger.error("Dataset type must be either 'train' or 'test'")
        raise ValueError("Dataset type must be either 'train' or 'test'")

    df = data.copy()
    logger.info(f'Data loaded successfully from {input_data_path}')

    # Feature_engineering for Time column
    #Log
    df['Time_log1p'] = np.log1p(df['Time'])

    #Buckets <1m <10m <1h <6h <24h >24h
    bins = time_bins
    labels = time_labels
    df['Time_buckets'] = pd.cut(df['Time'], bins=bins, labels=labels)

    #Feature_engineering for Amount column
    #Log
    df['Amount_log1p'] = np.log1p(df['Amount'])

    #Buckets <10 <100 <500 <1000 <5000 >5000
    bins = amount_bins
    labels = amount_labels
    df['Amount_buckets'] = pd.cut(df['Amount'], bins=bins, labels=labels)

    # Interaction feature between Amount and Time
    df['amount_time_interaction'] = df['Amount'] * df['Time']
    df['amount_time_interaction_log1p'] = np.log1p(df['amount_time_interaction'])

    # Interaction feature between V1 - V28
    feature_cols = [col for col in df.columns if col.startswith('V')]
    poly = PolynomialFeatures(degree=polynomial_degree , interaction_only=True, include_bias=False)
    interaction_terms = poly.fit_transform(df[feature_cols])
    interaction_feature_names = poly.get_feature_names_out(feature_cols)
    interaction_df = pd.DataFrame(interaction_terms, columns=interaction_feature_names)
    interaction_df = interaction_df.drop(columns=feature_cols)  # Drop original features to keep only interaction terms
    df = pd.concat([df, interaction_df], axis=1)

    # Save the dataframe with new features to a new CSV file
    df.to_csv(os.path.join(output_data_path,dataset+'_engineered.csv'), index=False)
    logger.info(f'Feature engineered data saved to {output_data_path}')


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),  # console
            logging.FileHandler("../../../logs/feature_engineering.log", encoding="utf-8"),  # file
        ],
        force=True,
    )

    loaded_config = load_config(logger=logger,yaml_file_name='feature_engineering.yaml')
    create_features(logger=logger, dataset ='train',**loaded_config)
    create_features(logger=logger, dataset ='test',**loaded_config)





