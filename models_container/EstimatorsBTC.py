import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
import yfinance as yf



class EstimatorsBTC:

    def __init__(self):

        self.X = ...
        self.y = ...
        
        # Estimators trained with the best hyperparameters found in the hyperparameter tuning process
        self.estimators = {
            'GradientBoosting': GradientBoostingClassifier(learning_rate=0.04,
                                                           max_depth=3,
                                                           n_estimators=110),

            'RandomForest': RandomForestClassifier(bootstrap=False,
                                                   max_depth=7,
                                                   n_estimators=250),

            'AdaBoost': AdaBoostClassifier(learning_rate=0.05, 
                                           n_estimators=30),

            'SVC': SVC(kernel="linear",
                       C=1100,
                       gamma=0.05,
                       probability=True)
        }

        # Initial performances measured with the best hyperparameters found in the hyperparameter tuning process (Real-Time-Scenario CV)
        self.performances = {
            'GradientBoosting': {"recall": 0.4, "precision": 0.89},
            'RandomForest': {"recall": 0.31, "precision": 0.89},
            'AdaBoost': {"recall": 0.27, "precision": 1.0},
            'SVC': {"recall": 0.94, "precision": 1.0},
        }


    def _fit_estimators(self) -> None:
        


