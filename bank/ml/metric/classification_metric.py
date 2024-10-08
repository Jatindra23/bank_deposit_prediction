from bank.entity.artifact_entity import ClassificationMetricArtifact
from bank.exception import BankException
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import os, sys


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        # model_precision_score = precision_score(y_true, y_pred)
        # model_recall_score = recall_score(y_true, y_pred)
        moodel_accuracy_score = accuracy_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(
            accuracy_score=moodel_accuracy_score,
            f1_score=model_f1_score,
            # precision_score=model_precision_score,
            # recall_score=model_recall_score,
        )

        return classification_metric
    except Exception as e:
        raise BankException(e, sys)
