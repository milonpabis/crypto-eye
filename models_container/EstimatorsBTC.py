import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
import yfinance as yf

from feature_generator.FeatureGenerator import FeatureGenerator



class EstimatorsBTC:

    def __init__(self):

        self.X: np.ndarray
        self.y: np.ndarray
        self.Xtoday: np.ndarray

        self.features = ["RSI5", "RSI7", "RSI14", "RSI20", "CCI3", "CCI5", "CCI7", "CCI14", "CCI20", "SOMA37", "SOMA314", "MACD"]
        
        # Estimators trained with the best hyperparameters found in the hyperparameter tuning process
        self.estimators = {
            'GradientBoost': {"estimator": GradientBoostingClassifier(learning_rate=0.1,
                                                                         max_depth=3,
                                                                         n_estimators=100),
                                 "threshold": 0.53,
                                      "type": "anchored"},

            'RandomForest': {"estimator": RandomForestClassifier(bootstrap=False,
                                                                 max_depth=21,
                                                                 n_estimators=160), 
                             "threshold": 0.55,
                                  "type": "anchored"},

            'AdaBoost': {"estimator": AdaBoostClassifier(algorithm="SAMME"),
                         "threshold": 0.52,
                              "type": "anchored"},
        }

        # Initial performances measured with the best hyperparameters found in the hyperparameter tuning process (Real-Time-Scenario CV)
        self.performances = {
            'GradientBoost': {"recall": 0.51, "precision": 0.65},
            'RandomForest': {"recall": 0.45, "precision": 0.63},
            'AdaBoost': {"recall": 0.76, "precision": 0.64}
        }


        self._load_data()
        self._fit_estimators()


    def _load_data(self) -> None:
        bitcoin = yf.Ticker("BTC-USD")
        data = bitcoin.history(period="max")
        self.X, self.y, self.today = FeatureGenerator.generate_features(data, HLC_targets=["High", "Low", "Close"], features=self.features, output_name="Growth")


    def _fit_estimators(self) -> None:
        for est in self.estimators:
            self.estimators[est]["estimator"].fit(self.X, self.y)
            print(f"{est} fitted")

    
    def predict_today(self) -> None:
        for est in self.estimators:
            y_prob = self.estimators[est]["estimator"].predict_proba(self.today)[:,1]
            print(f"{est} predicted: {y_prob > self.estimators[est]['threshold']}")

        


