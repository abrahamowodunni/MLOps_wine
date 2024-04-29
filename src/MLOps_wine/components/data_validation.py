import os
from src.MLOps_wine import logger
import pandas as pd
from src.MLOps_wine.entity.config_entity import DataValidationConfig

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    
    def validate_all_columns(self)-> bool:
        try:
            validation_status = None # we are creating of process of checking 

            data = pd.read_csv(self.config.unzip_data_dir) # read in the data
            all_cols = list(data.columns) # checking the columns

            all_schema = self.config.all_schema.keys() # pulling from our schema.yaml 

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e