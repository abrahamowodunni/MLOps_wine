import os
import pandas as pd
from src.MLOps_wine import logger
from src.MLOps_wine.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import joblib
from src.MLOps_wine.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,true, predicted):
        accuracy = accuracy_score(true, predicted)
        precision = precision_score(true, predicted, average='weighted')  # Update here
        recall = recall_score(true, predicted, average='weighted')  # Update here
        f1 = f1_score(true, predicted, average='weighted')  # Update here
        return accuracy, precision, recall, f1

    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        label_mapping = {'low': 0, 'medium': 1, 'high': 2}
        test_data[self.config.target_column] = test_data[self.config.target_column].map(label_mapping)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]
        
        predicted_qualities = model.predict(test_x)

        (accuracy, precision, recall, f1) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1':f1}
        save_json(path=Path(self.config.metric_file_name), data=scores)