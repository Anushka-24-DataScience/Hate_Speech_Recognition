import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate_speech.logger import logging
from hate_speech.exception import CustomException
from keras.utils import pad_sequences
from hate_speech.constants import *
from sklearn.metrics import confusion_matrix
from hate_speech.entity.config_entity import ModelEvaluationConfig
from hate_speech.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        :param model_evaluation_config: Configuration for model evaluation
        :param model_trainer_artifacts: Output reference of model trainer artifact stage
        """

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def evaluate(self):
        """
        :return: model accuracy
        """
        try:
            logging.info("Entering the evaluate function of Model Evaluation class")

            # Load test data
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            # Load tokenizer
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # Load model
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            # Prepare test data
            x_test = x_test['tweet'].astype(str).squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            # Evaluate model
            _, accuracy = load_model.evaluate(test_sequences_matrix, y_test, verbose=0)
            logging.info(f"Test accuracy: {accuracy * 100:.2f}%")

            # Confusion matrix
            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = [1 if pred[0] >= 0.5 else 0 for pred in lstm_prediction]
            logging.info(f"Confusion matrix: \n{confusion_matrix(y_test, res)}")

            return accuracy * 100  # Returning percentage accuracy
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Method to initiate model evaluation and check if the trained model's accuracy is acceptable.
        """
        logging.info("Initiating model evaluation")
        try:
            # Evaluate the current trained model
            trained_model_accuracy = self.evaluate()

            # Accept the model only if the accuracy is greater than 80%
            if trained_model_accuracy >= 80:
                is_model_accepted = True
                logging.info(f"Trained model is accepted with accuracy: {trained_model_accuracy:.2f}%")
            else:
                is_model_accepted = False
                logging.info(f"Trained model is rejected with accuracy: {trained_model_accuracy:.2f}%")

            # Create ModelEvaluationArtifacts
            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
