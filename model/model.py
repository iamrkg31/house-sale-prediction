import pickle
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

class MODEL(object):
    def __init__(self, X_train=None, Y_train=None, X_dev=None, Y_dev=None, X_test=None, Y_test=None):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_dev = X_dev
        self.Y_dev = Y_dev
        self.X_test = X_test
        self.Y_test = Y_test

    def regressor(self, name=None, path_model=None,  is_test=True):
        if not is_test:
            switcher = {
                "LinearRegressor": LinearRegression(),
                "RandomForestRegressor" : RandomForestRegressor(random_state=0),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),
            }
            model = switcher[name]
            model.fit(self.X_train, self.Y_train)
            Y_pred = model.predict(self.X_dev)
            score = r2_score(self.Y_dev, Y_pred)
            print(name, ' with R2 score {} done.'.format(score))
            pickle.dump(model, open(path_model, 'wb'))

        else:
            model = pickle.load(open(path_model, 'rb'))
            Y_pred = model.predict(self.X_test)
            return Y_pred

