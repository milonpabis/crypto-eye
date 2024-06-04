from models_container.EstimatorsBTC import EstimatorsBTC
import datetime as dt

if __name__ == "__main__":
    print("--------- Loading and preprocessing data ---------")
    start = dt.datetime.now()
    btc = EstimatorsBTC()
    btc._load_data()
    end = dt.datetime.now()
    print(f"--------- Time taken: {end-start} ---------")