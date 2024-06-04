import pandas as pd
import numpy as np


class FeatureGenerator:


    @staticmethod
    def RSI(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


    @staticmethod
    def generate_features(data: pd.DataFrame, target: str, output_name: str = "Growth") -> pd.DataFrame:
        features = ["Open", "Close", "Volume", "1D", "30D", "200DA", "200DEA", "3RSI", "7RSI", "14RSI", "30RSI", "50RSI", "200RSI"]
        return_data = data.copy()
        past_prices_values = [1, 30]
        MA_values = [200]
        RSI_values = [3, 7, 14, 30, 50, 200]

        for pp in past_prices_values:
             # PAST PRICES
             return_data[f"{pp}D"] = return_data[target].shift(pp)

        for window in MA_values:
            # MOVING AVERAGE
            return_data[f"{window}DA"] = return_data[target].rolling(window=window).mean()

            # EXPONENTIAL MOVING AVERAGE
            return_data[f"{window}DEA"] = return_data[target].ewm(span=window, adjust=False).mean()

        for window in RSI_values:
            return_data[f"{window}RSI"] = FeatureGenerator.RSI(return_data[target].copy(), window=window)

        
        return_data[output_name] = (return_data[target] > return_data["1D"]).astype(int)
        return_data.dropna(inplace=True)
        
        return return_data[features].values, return_data[output_name].values

        


