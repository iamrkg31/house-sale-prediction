import pickle
import numpy as np
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


    def dnn_regressor(self):
        steps = int(len(self.X_train)/128)
        feature_col = [tf.feature_column.numeric_column(key='feature',shape=(self.X_train.shape[1],))]
        model = tf.contrib.learn.DNNRegressor(hidden_units=[1024, 512, 256],
                        optimizer=tf.train.ProximalAdagradOptimizer(
                          learning_rate=0.005,
                          l1_regularization_strength=0.001
                        ), feature_columns=feature_col)
        for i in range(1000):
            model.fit(input_fn=tf.estimator.inputs.numpy_input_fn(
                dict({'feature': self.X_train}), self.Y_train,
                shuffle=True), steps=steps)
        predictions = model.predict_scores(input_fn=tf.estimator.inputs.numpy_input_fn(dict({'feature': self.X_dev}), self.Y_dev, batch_size=len(self.Y_dev), shuffle=False), as_iterable=False)

        print_rmse(model, 'eval', input_fn=tf.estimator.inputs.numpy_input_fn(dict({'feature': self.X_dev}), self.Y_dev, num_epochs=1, shuffle=False))

        score = r2_score(self.Y_dev, np.array(list(predictions)))
        print(score)

        for i in range(4323):
            print(self.Y_dev[i], " : ", predictions[i])



def print_rmse(model, name, input_fn):
  metrics = model.evaluate(input_fn=input_fn, steps=1)
  print(metrics)
  # print ('RMSE on {} dataset = {} USD'.format(name, np.sqrt(metrics['average_loss'])))

