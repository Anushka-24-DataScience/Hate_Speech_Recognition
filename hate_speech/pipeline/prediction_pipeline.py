import os
import sys
import keras
import pickle
from hate_speech.logger import logging
from hate_speech.constants import *
from hate_speech.exception import CustomException
from keras.utils import pad_sequences
from hate_speech.components.data_transforamation import DataTransformation
from hate_speech.entity.config_entity import DataTransformationConfig
from hate_speech.entity.artifact_entity import DataIngestionArtifacts

class PredictionPipeline:
    def __init__(self):
        # Local model path instead of GCloud
        self.model_path = os.path.join("artifacts", "PredictModel", MODEL_NAME)
        self.data_transformation = DataTransformation(data_transformation_config=DataTransformationConfig,
                                                      data_ingestion_artifacts=DataIngestionArtifacts)

    def predict(self, text):
        """load model and tokenizer, then make predictions"""
        logging.info("Running the predict function")
        try:
            # Load the model from the local path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

            load_model = keras.models.load_model(self.model_path)
            
            # Load the tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            # Clean and preprocess the input text
            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]  # Tokenizer expects a list of texts
            
            # Tokenize and pad the sequence
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)

            # Make predictions
            pred = load_model.predict(padded)
            print(f"Prediction: {pred}")

            if pred > 0.5:
                logging.info("Prediction: hate and abusive")
                return "hate and abusive"
            else:
                logging.info("Prediction: no hate")
                return "no hate"

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self, text):
        """Run the prediction pipeline"""
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:
            predicted_text = self.predict(text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e
