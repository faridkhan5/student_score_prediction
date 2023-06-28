import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
#we give input to the data_ingestion comp
#because of this config, the comp knows where to save train, test and raw data
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')
    #artifacts is a dir used as loc to save the output files generate during data ingestion process - makes it easier to manage and access data


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        #initializes an instance of DataIngestionConfig clas
        #the 3 file paths (train, test, raw) will get saved in class var

    def initiate_data_ingestion(self):
    #begins data ingestion process
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            #reads the dataset from the source
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            #creates necessary dirs for the file paths specified in DataIngestionConfig object

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of data is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()


