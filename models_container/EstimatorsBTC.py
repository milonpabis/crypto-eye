import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
import yfinance as yf

from feature_generator.FeatureGenerator import FeatureGenerator



class EstimatorsBTC:

    def __init__(self):

        self.X: np.ndarray
        self.y: np.ndarray
        
        # Estimators trained with the best hyperparameters found in the hyperparameter tuning process
        self.estimators = {
            'GradientBoosting': GradientBoostingClassifier(learning_rate=0.04,
                                                           max_depth=3,
                                                           n_estimators=110),

            'RandomForest': {"estimator": RandomForestClassifier(bootstrap=False,
                                                                 max_depth=6,
                                                                 n_estimators=100), 
                             "threshold": 0.53,
                                  "type": "anchored"},

            'AdaBoost': {"estimator": AdaBoostClassifier(learning_rate=1, 
                                           n_estimators=500),
                         "threshold": 0.5,
                              "type": "anchored"},

            'SVC': {"estimator": SVC(kernel="rbf",
                                     C=5,
                                     gamma=1,
                                     probability=True),
                    "threshold": 0.52,
                         "type": "anchored"}
        }

        # Initial performances measured with the best hyperparameters found in the hyperparameter tuning process (Real-Time-Scenario CV)
        self.performances = {
            'GradientBoosting': {"recall": 0.4, "precision": 0.89},
            'RandomForest': {"recall": 0.18, "precision": 0.56},
            'AdaBoost': {"recall": 0.49, "precision": 0.62},
            'SVC': {"recall": 0.39, "precision": 0.51},
        }


    def _fit_estimators(self) -> None:
        pass


    def _load_data(self) -> None:
        bitcoin = yf.Ticker("BTC-USD")
        data = bitcoin.history(period="max")
        self.X, self.y = FeatureGenerator.generate_features(data, "Close")


    def _fit_models(self) -> None:
        
        for est in self.estimators:
            self.estimators[est].fit(self.X, self.y)
            print(f"{est} fitted")

        


