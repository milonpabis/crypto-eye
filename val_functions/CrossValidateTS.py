import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
from functools import partial


class CrossValidateTS:
    """
    Class containing useful functions to perform cross-validation on time series data.
    """

    @staticmethod
    def prediction_scorer_threshold(model: object, X: np.array, y: np.array, threshold: float = 0.6, scoring=precision_score) -> float:
        """
        Calculates the given metric, classifying the class 1 under the condition of exceeding the given probability threshold.

        Parameters
        ----------
        model : object
            Model to be used for cross-validation. Must have a predict_proba method.

        X : np.array
            Features of the dataset.

        y : np.array
            Target variable of the dataset.

        threshold : float, optional
            Probability threshold to classify the class 1. Default is 0.6.

        scoring : function, optional
            Scoring function to be used. Default is precision_score.

        Returns
        -------
        float
            The value of the metric.
        """

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob > threshold).astype(int)

        return scoring(y, y_pred, pos_label=1, zero_division=0)
    

    @staticmethod
    def check_model_awf(model: object, X: np.array, y: np.array, splits: int = 5, threshold: float = 0.6) -> float:
        """
        Cross-validates the Time Series data with Anchored Walking Forward approach. Creates a prediction for more than 1 future observation.

        Parameters
        ----------
        model : object
            The machine learning model to be cross-validated.

        X : np.array
            The input features for the model.

        y : np.array
            The target variable for the model.

        splits : int, optional
            The number of splits for the time series cross-validation. Default is 5.

        threshold : float, optional
            The threshold value for the prediction scorer. Default is 0.6.

        Returns
        -------
        float
            The mean score of the cross-validation.
        """
        cvts = TimeSeriesSplit(n_splits=splits)
        output = cross_validate(model, X, y, cv=cvts, scoring={"Precision": partial(CrossValidateTS.prediction_scorer_threshold, threshold=threshold),
                                                            "Recall": partial(CrossValidateTS.prediction_scorer_threshold, threshold=threshold, scoring=recall_score)})

        return np.mean(output["test_Precision"]), np.mean(output["test_Recall"])
    

    @staticmethod
    def check_model_rwf(model: object, X: np.array, y: np.array, max_train_size: int = 500, test_size: int = 100, threshold: float = 0.6) -> float:
        """
        Cross-validates the Time Series data with Rolling Walking Forward approach.

        Parameters
        ----------
        model : object
            The machine learning model to be cross-validated.

        X : np.array
            The input features.

        y : np.array
            The target variable.

        max_train_size : int, optional
            The maximum size of the training set. Default is 500.

        test_size : int, optional
            The size of each test set. Default is 100.

        threshold : float, optional
            The threshold value for prediction scoring. Default is 0.6.

        Returns
        -------
        float
            The mean score of the cross-validation.

        """
        splits = (len(X)-max_train_size)//test_size
        cvts = TimeSeriesSplit(n_splits=splits, max_train_size=max_train_size, test_size=test_size)
        output = cross_validate(model, X, y, cv=cvts, scoring={"Precision": partial(CrossValidateTS.prediction_scorer_threshold, threshold=threshold),
                                                            "Recall": partial(CrossValidateTS.prediction_scorer_threshold, threshold=threshold, scoring=recall_score)})

        return np.mean(output["test_Precision"]), np.mean(output["test_Recall"])
    

    @staticmethod
    def cross_validate_rts(model: object, X: np.array, y: np.array, threshold: float = 0.6, n_days: int = 150):
        """
        Creates a Real-Time-Scenario backtesting cross-validation. 
        Training the model with the training set and predicting the test set of 1 observation.

        Parameters
        ----------
        model : object
            The machine learning model to be used for training and prediction.

        X : np.array
            The input features for training and prediction.

        y : np.array
            The target variable for training and prediction.

        threshold : float, optional
            The threshold value for classification. Default is 0.6.

        n_Days : int, optional
            The size of the simulation. Default is 150.

        Returns
        -------
        y_test : np.array
            The actual target values for the test set.

        y_pred : np.array
            The predicted target values for the test set.
        """
        n_rows = len(X)
        cv = TimeSeriesSplit(max_train_size=n_rows-n_days, test_size=1, n_splits=n_days)
        y_test = []
        y_pred = []
        for train_id, test_id in cv.split(X):
            model.fit(X[train_id], y[train_id])
            y_prob = model.predict_proba(X[test_id])[:,1]
            y_pred.append(y_prob > threshold)
            y_test.append(y[test_id])
        return np.ravel(np.array(y_test)), np.ravel(np.array(y_pred))
    

    @staticmethod
    def cross_validate_rts_na(model: object, X: np.array, y: np.array, threshold: float = 0.6, n_days: int = 150):
        """
        Creates a Real-Time-Scenario backtesting cross-validation that is NON ANCHORED (ROLLING). 
        Training the model with the training set and predicting the test set of 1 observation.

        Parameters
        ----------
        model : object
            The machine learning model to be used for training and prediction.

        X : np.array
            The input features for training and prediction.

        y : np.array
            The target variable for training and prediction.

        threshold : float, optional
            The threshold value for classification. Default is 0.6.

        n_Days : int, optional
            The size of the simulation. Default is 150.

        Returns
        -------
        y_test : np.array
            The actual target values for the test set.

        y_pred : np.array
            The predicted target values for the test set.
        """
        cv = TimeSeriesSplit(max_train_size=500, test_size=1, n_splits=n_days)
        y_test = []
        y_pred = []
        for train_id, test_id in cv.split(X):
            model.fit(X[train_id], y[train_id])
            y_prob = model.predict_proba(X[test_id])[:,1]
            y_pred.append(y_prob > threshold)
            y_test.append(y[test_id])
        return np.ravel(np.array(y_test)), np.ravel(np.array(y_pred))
    
    
    @staticmethod
    def find_best_threshold(model, X: np.ndarray, y: np.ndarray, min_threshold: float = 0.5, max_threshold: float = 0.6, step: float = 0.01,
                                                             awf_splits: int = 5, rwf_max_train_size: int = 500, rwf_test_size: int = 100) -> None:
        """
        Prints the model performance with both - anchored and rolling walking forward CVs, for different thresholds.
        """
        
        for thr in np.arange(min_threshold, max_threshold+step, step):
            a_precision, a_recall = CrossValidateTS.check_model_awf(model, X, y, splits=awf_splits, threshold=thr)
            r_precision, r_recall = CrossValidateTS.check_model_rwf(model, X, y, max_train_size=rwf_max_train_size, test_size=rwf_test_size, threshold=thr)
            print(f"-*-*- THRESHOLD = {round(thr, 2)} -*-*-")
            print(f"ANCHORED --- | PRECISION = {a_precision} | RECALL = {a_recall}")
            print(f"ROLLING --- | PRECISION = {r_precision} | RECALL = {r_recall}")
            print("\n\n")
    

    
