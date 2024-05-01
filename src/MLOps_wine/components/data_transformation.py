import os
from src.MLOps_wine import logger
from src.MLOps_wine.entity.config_entity import DataTransformationConfig

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from joblib import dump


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        train,test = train_test_split(data,test_size=self.config.test_size)
        bins = self.config.bins  # Quality bins
        labels = self.config.labels  # Quality labels

        data[self.config.target_column] = pd.cut(data[self.config.target_column],bins=bins, labels=labels, include_lowest=True)

        numeric_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        numeric_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            [
                ("StandardScaler", numeric_transformer, numeric_features)
            ]
        )

        # Fit preprocessor on training data and transform both train and test data
        train = preprocessor.fit_transform(train)
        test = preprocessor.transform(test)

        # Convert transformed arrays back to DataFrame
        train = pd.DataFrame(train, columns=numeric_features)
        test = pd.DataFrame(test, columns=numeric_features)

        # Concatenate the target column to the DataFrame
        train[self.config.target_column] = data.loc[train.index, self.config.target_column]
        test[self.config.target_column] = data.loc[test.index, self.config.target_column]

        # Save preprocessed data to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        dump(preprocessor, os.path.join(self.config.root_dir, "preprocessor.joblib"))

        logger.info(f"Data split into training and test sets (test_size: {self.config.test_size})")
        logger.info(f"Training features shape: {train.shape}")
        logger.info(f"Test features shape: {test.shape}")

        print(f"Data split into training and test sets (test_size: {self.config.test_size})")
        print(f"Training features shape: {train.shape}")
        print(f"Test features shape: {test.shape}")
        print(f'number of columns {numeric_features}')
    
