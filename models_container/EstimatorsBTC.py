import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional

from feature_generator.FeatureGenerator import FeatureGenerator
from model_tracking.DataBaseLogs import DBLogs

DEBUG = True
DATE_FORMAT = r"%Y-%m-%d"


class EstimatorsBTC:

    def __init__(self):

        self.X: np.ndarray
        self.y: np.ndarray
        self.Xtoday: np.ndarray

        self.modelDB = DBLogs()

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

        if not DEBUG:
            self._initialize_estimators(go_back=0)


    def _load_data(self, max_date: str = datetime.now().strftime(DATE_FORMAT), retrieve: bool = False) -> Optional[tuple]:
        bitcoin = yf.Ticker("BTC-USD")
        data = bitcoin.history(start=None, end=max_date, period="max")
        X, y, Xtoday = FeatureGenerator.generate_features(data, HLC_targets=["High", "Low", "Close"], features=self.features, output_name="Growth")

        self.X = X.values
        self.y = np.ravel(y.values)
        self.Xtoday = np.atleast_2d(Xtoday.values)
        if retrieve:
            return X, y, Xtoday



    def _fit_estimators(self) -> None:
        for est in self.estimators:
            self.estimators[est]["estimator"].fit(self.X, self.y)
            print(f"{est} fitted")

    
    def predict_today(self) -> dict:
        results = {}
        for est in self.estimators:
            y_prob = self.estimators[est]["estimator"].predict_proba(self.Xtoday)[:,1]
            y_pred = int((y_prob > self.estimators[est]["threshold"])[0])
            print(f"{est} predicted: {y_pred}")
            results[est] = y_pred
        return results


    def _initialize_estimators(self, max_date: str = datetime.now().strftime(DATE_FORMAT)) -> None:
        self._load_data(max_date=max_date)
        self._fit_estimators()

    
    def update_performance(self, days_back: int = 150) -> None:
        """
        Updates the missing performance values for the estimators for the last 150 days.
        Checks on which days the performance and the predictions are missing and performs the backtesting evaluation.

        This method should be called only in order to keep the performance values updated.
        
        !WARNING!
        It may take a long time to run, depending on the number of days to be evaluated.
        """
        
        today = datetime.now().strftime(DATE_FORMAT)
        today150 = (datetime.now() - timedelta(days=days_back)).strftime(DATE_FORMAT)

        missing_dates = self.modelDB.get_missing_dates(today150, today)
        
        for date in missing_dates:
            print(f"""Evaluating date: {date}""")
            self._initialize_estimators(max_date=date)
            res = self.predict_today()
            for est in res:
                self.modelDB.insert_model_prediction(est, date, res[est])

        self.fill_real_predictions(today150, today)


    def fill_real_predictions(self, start_date: str, end_date: str) -> None:
        """
        Fills all the missing real predictions for the last 150 days.
        """
        dates = self.modelDB.get_missing_dates(start_date=start_date,
                                                end_date=end_date,
                                                  difference=False)

        _, y, _ = self._load_data(retrieve=True)

        for date in dates:
            y_true = y[y.index == date].values[0][0]
            self.modelDB.insert_real_value(y_true=y_true, date=date)

        
            







        


