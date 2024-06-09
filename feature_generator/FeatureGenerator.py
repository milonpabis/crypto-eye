import pandas as pd
import numpy as np
from typing import List


class FeatureGenerator:


    @staticmethod
    def RSI(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculates the Relative Strength Index (RSI) for a given series of data.

        Parameters:
            data (pd.Series): The input data series.
            window (int): The number of periods to consider for the RSI calculation. Default is 14.

        Returns:
            pd.Series: The RSI values for the input data series.
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    

    @staticmethod
    def MACD_hist(data: pd.Series) -> pd.Series:
        """
        Calculates the MACD histogram for the given data.

        Parameters:
            data (pd.Series): The input data for which MACD histogram needs to be calculated.

        Returns:
            pd.Series: The MACD histogram values.
        """
        EMA12 = data.ewm(span=12, adjust=False).mean()
        EMA26 = data.ewm(span=26, adjust=False).mean()
        
        MACD = EMA12 - EMA26
        signal_line = MACD.ewm(span=9, adjust=False).mean()

        MACD_HIST = MACD - signal_line
        MAX_MACD = MACD_HIST.max()
        MIN_MACD = MACD_HIST.min()

        # I also return the MAX_MACD and MIN_MACD for the training data in order to deal with extrapolation on test data
        return MACD_HIST, MAX_MACD, MIN_MACD
    

    @staticmethod
    def CCI(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculates the Commodity Channel Index (CCI) for a given set of high, low, and close prices.

        Parameters:
            high (pd.Series): Series containing the high prices.
            low (pd.Series): Series containing the low prices.
            close (pd.Series): Series containing the close prices.
            window (int): The window size for calculating the moving average (default: 20).

        Returns:
            pd.Series: Series containing the calculated CCI values.
        """
        TP = (high + low + close) / 3
        MA = TP.rolling(window=window).mean()
            
        CCI = ((TP - MA) / (0.015 * np.mean(np.absolute(TP - np.mean(TP)))))
        return CCI
    

    @staticmethod
    def stochastic_oscilator(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculates the stochastic oscillator for a given data series.

        Parameters:
            data (pd.Series): The input data series.
            window (int): The window size for calculating the oscillator. Default is 14.

        Returns:
            pd.Series: The calculated stochastic oscillator values.
        """
        last_x_max = data.rolling(window).max()
        last_x_min = data.rolling(window).min()

        return 100 * (data - last_x_min) / (last_x_max - last_x_min)


    @staticmethod
    def generate_features(data: pd.DataFrame, features: List[str], HLC_targets: List[str] = ["High", "Low", "Close"],  output_name: str = "Growth") -> pd.DataFrame:
        """
        Generate features for stock data.

        Parameters:
            data (pd.DataFrame): The input stock data.
            features (List[str]): List of feature names to include in dataset.
            HLC_targets (List[str], optional): List of column names for High, Low, and Close targets. Defaults to ["High", "Low", "Close"].
            output_name (str, optional): Name of the output variable. Defaults to "Growth".

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing X_train, y_train, and X_today.
                - X_train (np.ndarray): Training features.
                - y_train (np.ndarray): Training target variable.
                - X_today (np.ndarray): Features for the current day.
        """
        RSI_windows = [5, 7, 14, 20]
        CCI_windows = [3, 5, 7, 14, 20]
        SO_windows = [7, 14]

        for window in RSI_windows:
            data[f"RSI{window}"] = FeatureGenerator.RSI(data[HLC_targets[2]], window=window)

        for window in CCI_windows:
            data[f"CCI{window}"] = FeatureGenerator.CCI(data[HLC_targets[0]], data[HLC_targets[1]], data[HLC_targets[2]], window=window)

        for window in SO_windows:
            data[f"SO{window}"] = FeatureGenerator.stochastic_oscilator(data[HLC_targets[2]], window=window)
            data[f"SOMA3{window}"] = data[f"SO{window}"].rolling(3).mean()

        data["MACD"] = FeatureGenerator.MACD_hist(data[HLC_targets[2]])[0]

        # Target variable
        data[output_name] = (data[HLC_targets[2]] < data[HLC_targets[2]].shift(-1)).astype(int)

        data.dropna(inplace=True)

               # X_train                           y_train                                            X_today
        return data[features].iloc[:-1, :].values, np.ravel(data[[output_name]].iloc[:-1, :].values), np.atleast_2d(data[features].iloc[-1, :].values)



        


