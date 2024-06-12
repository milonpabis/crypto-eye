import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score
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
            self.__initialize_estimators()
            res = self.predict_today()
            for est in res:
                self.modelDB.insert_model_prediction(est, datetime.now().strftime(DATE_FORMAT), res[est])
            self.fill_real_predictions(start_date=None, end_date=None)



    def predict_today(self) -> dict:
        """
        Predicts the potential growth for current self.Xtoday values by every estimator.
        """
        results = {}    # dictionary to store the results
        for est in self.estimators: 
            y_prob = self.estimators[est]["estimator"].predict_proba(self.Xtoday)[:,1]
            y_pred = int((y_prob > self.estimators[est]["threshold"])[0])
            results[est] = y_pred
        return results


    
    def update_predictions(self, days_back: int = 150) -> None:
        """
        Updates the missing prediction values for the estimators for the last 150 days.
        Checks on which days the predictions are missing and performs the backtesting evaluation.

        This method should be called only in order to keep the prediction values updated.
        
        !WARNING!
        It may take a long time to run, depending on the number of days to be evaluated.
        """
        
        today = datetime.now().strftime(DATE_FORMAT)
        today_nback = (datetime.now() - timedelta(days=days_back)).strftime(DATE_FORMAT)    # date days_back ago

        missing_dates = self.modelDB.get_missing_dates_predictions(today_nback, today)  # getting the missing prediction dates for the last days_back days
        
        for date in missing_dates:  # for every of these dates, make a prediction and store it in the database
            print(f"""Evaluating date: {date}""")
            self.__initialize_estimators(max_date=date)   # gets the data for historical dates and fits the estimators
            res = self.predict_today()  # predicts for self.Xtoday
            for est in res:
                self.modelDB.insert_model_prediction(est, date, res[est])   # for every estimator in res(dict), insert into the db

        self.fill_real_predictions(start_date=None, end_date=None)  # fills all the missing real values that are available in the database and yahoo finance


    
    def update_performance(self, estimator: str) -> None:
        data = self.modelDB.get_model_predictions(estimator) # getting the predictions from the database

        data = data.sort_values(by="date", ascending=True).dropna() # sorting the data by date

        missing_dates = self.modelDB.get_missing_dates_performance(estimator) # getting the missing performance dates for the estimator

        date150 = data["date"].iloc[149] # getting the date that thresholds the first 150 days
        date_range = np.ravel(missing_dates[missing_dates["date"] >= date150].values) # getting the missing dates that are after the date150

        for d in date_range:   # rolling window
            print(d)
            values150 = data[data["date"] <= d][["y_true", "y_pred"]].dropna().values # getting the values from before the date d
            # calculating metrics
            precision = precision_score(values150[:,0], values150[:,1])
            recall = recall_score(values150[:,0], values150[:,1])
            self.modelDB.insert_model_performance(estimator, d, recall, precision)  # adding the performance to the database



    def fill_real_predictions(self, start_date: str, end_date: str) -> None:
        """
        Fills all the missing real predictions for the last 150 days.
        """
        dates = self.modelDB.get_missing_dates_predictions(start_date=start_date,   # returns dates that are EXISTING in database, but have NULL y_true value
                                                end_date=end_date,
                                                  difference=False)

        _, y, _ = self.__load_data(retrieve=True)   # returns the pandas dataframes for X, y, Xtoday,
                                                    # also sets the self.X, self.y, self.Xtoday for the most recent values

        for date in dates:      # fill the missing values with the real ones from the y dataframe
            try:
                y_true = y[y.index == date].values[0][0]
                self.modelDB.insert_real_value(y_true=y_true, date=date)    # db update
            except:
                print("Skipped date: ", date)   # it is going to skip the today's date (because we dont know the result yet),
                                                # and the missing ones from yahoo finance



    def __initialize_estimators(self, max_date: str = None) -> None:
        """
        Loads the data with given time delay and fits the estimators.
        """
        self.__load_data(max_date=max_date) # loads the data for the most recent date
        self.__fit_estimators() # fits the estimators with that data



    def __load_data(self, max_date: str = None, retrieve: bool = False) -> Optional[tuple]:
        """
        Loads the data from Yahoo Finance API and generates the features.
        Also sets the self.X, self.y, self.Xtoday values with preprocessed data.

        Parameters:
        ----------
        max_date : str, optional
            The maximum date for the data. Default is None(=today).

        retrieve : bool, optional
            If True, returns the X, y, Xtoday values in pandas DataFrame. Default is False.
        """
        bitcoin = yf.Ticker("BTC-USD")
        data = bitcoin.history(start=None, end=max_date, period="max")

        X, y, Xtoday = FeatureGenerator.generate_features(data,
                                                          HLC_targets=["High", "Low", "Close"],
                                                          features=self.features,
                                                          output_name="Growth")

        self.X = X.values
        self.y = np.ravel(y.values)
        self.Xtoday = np.atleast_2d(Xtoday.values)

        if retrieve:
            return X, y, Xtoday
        


    def __fit_estimators(self) -> None:
        """
        Simply fits all the estimators with the current self.X, self.y values.
        """
        for est in self.estimators:
            self.estimators[est]["estimator"].fit(self.X, self.y)

        
            







        


