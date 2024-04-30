import pandas as pd
import os
from src.MLOps_wine import logger
from src.MLOps_wine.entity.config_entity import ModelTrainerConfig
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        oe = OrdinalEncoder()
        train_data[self.config.target_column] = oe.fit_transform(train_data[[self.config.target_column]])
        test_data[self.config.target_column] = oe.transform(test_data[[self.config.target_column]])


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        lr = LogisticRegression(C=self.config.C,solver=self.config.solver, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))