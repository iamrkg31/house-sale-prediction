import numpy as np
from config.config_parser import Config
from model.model import MODEL


conf = Config("config/system.config")
path_X_train = conf.get_config("PATHS", "path_X_train")
path_Y_train = conf.get_config("PATHS", "path_Y_train")
path_X_dev = conf.get_config("PATHS", "path_X_dev")
path_Y_dev = conf.get_config("PATHS", "path_Y_dev")
path_model_linear_regression = conf.get_config("PATHS", "path_model_linear_regression")

X_train = np.load(path_X_train)
Y_train = np.load(path_Y_train)
X_dev = np.load(path_X_dev)
Y_dev = np.load(path_Y_dev)

model = MODEL(X_train, Y_train, X_dev, Y_dev)
model.regressor("RandomForestRegressor", path_model_linear_regression, is_test=False)