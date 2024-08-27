from bank.entity.config_entity import TrainingPipelineConfig
from bank.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from bank.exception import BankException
from bank.entity.artifact_entity import DataIngestionArtifact
from bank.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
)

from bank.logger import logging
import sys
import os
from bank.components.data_ingestion import DataIngestion
from bank.components.data_validation import DataValidation
from bank.components.data_transformation import DataTransformation
from bank.components.model_trainer import ModelTrainer
from bank.components.model_evaluation import ModelEvaluation
from bank.components.model_pusher import ModelPusher


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:

            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"Data ingestion completed and artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise BankException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:

        try:

            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise BankException(e, sys)

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:

        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            return data_transformation_artifact

        except Exception as e:
            raise BankException(e, sys)

    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise BankException(e, sys)

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:

        try:
            model_evaluation_config = ModelEvaluationConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_evaluation = ModelEvaluation(
                model_evaluation_config=model_evaluation_config,
                data_validation_artifact=data_validation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact

        except Exception as e:
            raise BankException(e, sys)

    def start_model_pusher(
        self, model_evaluation_artifact: ModelEvaluationArtifact
    ) -> ModelPusherArtifact:

        try:

            model_pusher_config = ModelPusherConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_pusher = ModelPusher(
                model_pusher_config=model_pusher_config,
                model_evaluation_artifact=model_evaluation_artifact,
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact

        except Exception as e:
            raise BankException(e, sys)

    def run_pipeline(self):
        try:

            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
            logging.info("Data Ingestion Successfull")
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logging.info("Data Validation Successfull")

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            logging.info("Data Transformation Successfull")

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            logging.info("Model Trainer Successfull")

            model_evaluation_artifact = self.start_model_evaluation(
                data_validation_artifact,
                model_trainer_artifact,
            )
            logging.info("Model Evaluation Successfull")

            if not model_evaluation_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")

            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)

            logging.info("Model Pusher Successfull")

            TrainPipeline.is_pipeline_running = False

        except Exception as e:
            TrainPipeline.is_pipeline_running = False #if some exception occurs the above line before exception will not execute thats why this line in between Exception will execute
            raise BankException(e, sys)
