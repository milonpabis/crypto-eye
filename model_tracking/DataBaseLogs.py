import sqlite3
import os
import pandas as pd
from model_tracking.performance_data import PerformanceWindows, PerformanceBatch



class DBLogs:
    def __init__(self, db_path: str = "model_tracking\\data\\logs.db"):
        self.db_path = db_path


    def connect(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()


    def get_model_predictions(self, model_name: str) -> pd.DataFrame:
        """Returns the history of model predictions given its name."""
        try:
            model_id = self.get_model_id(model_name)
            return self.__get_model_predictions_id(model_id)
        
        except Exception as exception_error:
            print(exception_error)
            return None
        


    def get_model_prediction_date(self, model_name: str, date: str) -> int:
        """Returns the prediction value for a given date."""
        try:
            model_id = self.get_model_id(model_name)
            return self.__get_model_prediction_id_date(model_id, date)
        
        except Exception as exception_error:
            print(exception_error)
            return None
    


    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Returns the history of model performance given its name."""
        try:
            model_id = self.get_model_id(model_name)
            return self.__get_model_performance_id(model_id)
        
        except Exception as exception_error:
            print(exception_error)
            return None
        
    

    def insert_model_performance(self, performance_info: PerformanceWindows) -> None:
        """Inserts the current model performance into the database."""
        try:
            model_id = self.get_model_id(performance_info.get_estimator())
            self.__insert_performance_id(model_id, performance_info)

        except Exception as exception_error:
            print(exception_error)



    def insert_model_prediction(self, model_name: str, date: str, y_pred: int) -> None:
        """Inserts the current model prediction into the database. Does not include the real future value."""
        try:
            model_id = self.get_model_id(model_name)
            self.__insert_prediction_id(model_id, date, y_pred)

        except Exception as exception_error:
            print(exception_error)

    

    def insert_real_value(self, date: str, y_true: int) -> None:
        """Inserts the real value into the database."""
        try:
            self.cursor.execute("""UPDATE models_predictions SET y_true = ? WHERE date = ?;""", 
                (int(y_true), date))
            self.conn.commit()
        except Exception as exception_error:
            print(exception_error)



    def insert_model(self, model_name: str) -> None:
        """Inserts the model name into the database."""
        try:
            self.cursor.execute("""INSERT INTO models (model_name) VALUES (?);""", 
                                (model_name,))
            self.conn.commit()

        except Exception as exception_error:
            print(exception_error)



    def get_missing_dates_predictions(self, start_date: str, end_date: str, difference: bool = True) -> pd.DataFrame:
        """Returns the dates that are missing the predictions value."""
        try:
            if all([start_date, end_date]):
                self.cursor.execute("""SELECT DISTINCT date 
                                       FROM models_predictions 
                                       WHERE date BETWEEN ? AND ? AND y_pred IS NOT NULL;""",
                                    (start_date, end_date))
            else:
                self.cursor.execute("""SELECT DISTINCT date 
                                       FROM models_predictions 
                                       WHERE y_true IS NULL;""")
                
            existing_dates = pd.DataFrame(self.cursor.fetchall(), columns=["date"]).astype("datetime64[s]")

            if difference:
                dates150 = pd.date_range(start=start_date, end=end_date, inclusive="both")
                missing_dates = dates150[~dates150.isin(existing_dates["date"])].astype(str).str.split("T").str[0]
                return missing_dates.values
            
            return existing_dates["date"].astype(str).str.split("T").str[0].values
        
        except Exception as exception_error:
            print(exception_error)
            return None
        


    def get_missing_dates_performance(self, model_name: str) -> pd.DataFrame:
        model_id = self.get_model_id(model_name)
        self.cursor.execute("""
                            SELECT DISTINCT date 
                            FROM models_predictions
                            WHERE date NOT IN(
                                SELECT DISTINCT date
                                FROM models_performance
                                WHERE model_id = ?)
                            AND y_true IS NOT NULL;
                            """, (model_id,))
        
        return pd.DataFrame(self.cursor.fetchall(), columns=["date"])
        
    

    def create_tables(self):
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS models (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                model_name TEXT NOT NULL UNIQUE);
                            """)
        
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS models_performance (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                model_id INTEGER NOT NULL,
                                date TEXT NOT NULL,
                            
                                recall_total REAL NOT NULL,
                                precision_total REAL NOT NULL,
                                accuracy_total REAL NOT NULL,
                                specificity_total REAL NOT NULL,
                                neg_pred_value_total REAL NOT NULL,
                            
                                recall_7 REAL NOT NULL,
                                precision_7 REAL NOT NULL,
                                accuracy_7 REAL NOT NULL,
                                specificity_7 REAL NOT NULL,
                                neg_pred_value_7 REAL NOT NULL,
                            
                                recall_14 REAL NOT NULL,
                                precision_14 REAL NOT NULL,
                                accuracy_14 REAL NOT NULL,
                                specificity_14 REAL NOT NULL,
                                neg_pred_value_14 REAL NOT NULL,
                            
                                recall_30 REAL NOT NULL,
                                precision_30 REAL NOT NULL,
                                accuracy_30 REAL NOT NULL,
                                specificity_30 REAL NOT NULL,
                                neg_pred_value_30 REAL NOT NULL,

                                FOREIGN KEY (model_id) REFERENCES models (id),
                                UNIQUE (model_id, date));""")
        
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS models_predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                model_id INTEGER NOT NULL,
                                date TEXT NOT NULL,
                                y_true INTEGER,
                                y_pred INTEGER NOT NULL,
                                FOREIGN KEY (model_id) REFERENCES models (id),
                                UNIQUE (model_id, date));
                            """)
        
        self.conn.commit()



    def get_model_id(self, model_name: str) -> int:
        self.cursor.execute("""
                            SELECT id 
                            FROM models WHERE model_name = ?;
                            """, (model_name,))
        
        return self.cursor.fetchone()[0]
    


    def get_model_name(self, model_id: int) -> str:
        self.cursor.execute("""
                            SELECT model_name 
                            FROM models WHERE id = ?;
                            """, (model_id,))
        
        return self.cursor.fetchone()[0]
    


    def does_prediction_exists(self, date: str) -> bool:
        self.cursor.execute("""SELECT EXISTS(SELECT 1 FROM models_predictions WHERE date = ?);""", (date,))
        return self.cursor.fetchone()[0]
    
    

    def __insert_prediction_id(self, model_id: int, date: str, y_pred: int) -> None:
        self.cursor.execute("""
            INSERT INTO models_predictions (model_id, date, y_true, y_pred) VALUES (?, ?, NULL, ?);""", 
            (model_id, date, y_pred))
        
        self.conn.commit()
    


    def __insert_performance_id(self, model_id: int, performance_info: PerformanceWindows) -> None:
        self.cursor.execute("""
            INSERT INTO models_performance (model_id, date, recall_total, precision_total, accuracy_total, specificity_total, neg_pred_value_total,
                                            recall_7, precision_7, accuracy_7, specificity_7, neg_pred_value_7,
                                            recall_14, precision_14, accuracy_14, specificity_14, neg_pred_value_14,
                                            recall_30, precision_30, accuracy_30, specificity_30, neg_pred_value_30) 
                            
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""", 
            (model_id, performance_info.get_date(), *performance_info.get_data()))
        
        self.conn.commit()
    


    def __get_model_predictions_id(self, model_id: int) -> pd.DataFrame:
        self.cursor.execute("""
                            SELECT date, y_true, y_pred 
                            FROM models_predictions 
                            WHERE model_id = ?;
                            """, (model_id,))
        
        return pd.DataFrame(self.cursor.fetchall(), columns=["date", "y_true", "y_pred"])
    

    def __get_model_prediction_id_date(self, model_id: int, date: str) -> int:
        self.cursor.execute("""
                            SELECT y_pred
                            FROM models_predictions
                            WHERE model_id = ? AND date = ?;
                            """, (model_id, date))
        return self.cursor.fetchone()[0]
    
    

    def __get_model_performance_id(self, model_id: int) -> pd.DataFrame:
        self.cursor.execute("""
                            SELECT mp.date, m.model_name, mp.recall_total, mp.precision_total, mp.accuracy_total, mp.specificity_total, mp.neg_pred_value_total,
                                            mp.recall_7, mp.precision_7, mp.accuracy_7, mp.specificity_7, mp.neg_pred_value_7,
                                            mp.recall_14, mp.precision_14, mp.accuracy_14, mp.specificity_14, mp.neg_pred_value_14,
                                            mp.recall_30, mp.precision_30, mp.accuracy_30, mp.specificity_30, mp.neg_pred_value_30
                            FROM models_performance mp
                            JOIN models m
                            ON m.id = mp.model_id
                            WHERE mp.model_id = ?;
                            """, (model_id,))
        
        return pd.DataFrame(self.cursor.fetchall(), columns=PERFORMANCE_COLUMNS)
    


    def close(self):
        self.conn.close()





PERFORMANCE_COLUMNS = ["date", "model_name", "recall_total", "precision_total", "accuracy_total", "specificity_total", "neg_pred_value_total",
                       "recall_7", "precision_7", "accuracy_7", "specificity_7", "neg_pred_value_7",
                       "recall_14", "precision_14", "accuracy_14", "specificity_14", "neg_pred_value_14",
                       "recall_30", "precision_30", "accuracy_30", "specificity_30", "neg_pred_value_30"]

PREDICTION_COLUMNS = ["date", "y_true", "y_pred"]



if __name__ == "__main__": 
    db = DBLogs()
    db.insert_model("GradientBoost")
    db.insert_model("RandomForest")
    db.insert_model("AdaBoost")
    db.close()