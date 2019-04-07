import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from config.config_parser import Config


conf = Config("config/system.config")
path_data = conf.get_config("PATHS", "path_data")
path_X_train = conf.get_config("PATHS", "path_X_train")
path_Y_train = conf.get_config("PATHS", "path_Y_train")
path_X_dev = conf.get_config("PATHS", "path_X_dev")
path_Y_dev = conf.get_config("PATHS", "path_Y_dev")


def preprocess():
    df = pd.read_csv(path_data)
    df.drop(columns=["id", "sqft_lot", "sqft_lot15", "date", "zipcode", "condition", "floors", "yr_renovated"], axis=1, inplace=True)
    # df = remove_outliers(df)
    Y = df["price"].values
    df.drop(columns=["price"], axis=1, inplace=True)
    X = df.values
    X = preprocessing.MinMaxScaler().fit_transform(X)
    return X, Y

def remove_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    df = df[~((df < (q1 - 3 * iqr)) | (df > (q3 + 3 * iqr))).any(axis=1)]
    return df


def generate_train_test_data():
    X, Y = preprocess()
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=.20,
                                                        random_state=0)
    np.save(path_X_train, X_train)
    np.save(path_Y_train, Y_train)
    np.save(path_X_dev, X_test)
    np.save(path_Y_dev, Y_test)


generate_train_test_data()