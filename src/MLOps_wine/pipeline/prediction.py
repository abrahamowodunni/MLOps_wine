import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))
    
    def predict(self, data):
        preprocessed_data = self.preprocessor.transform(data)
        prediction = self.model.predict(preprocessed_data)

        return prediction