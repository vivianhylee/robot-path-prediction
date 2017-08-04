import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler


class GBM(object):
    def __init__(self, train_x, train_y, n_estimators, nodes):
        self.train_x = train_x
        self.train_y = train_y
        self.n_estimators = n_estimators
        self.nodes = nodes
        self.model = self.fit_gbm()

    def fit_gbm(self):
        model = MultiOutputRegressor(
            GradientBoostingRegressor(loss='ls', n_estimators=self.n_estimators, max_leaf_nodes=self.nodes))
        model.fit(self.train_x, self.train_y)
        return model

    def forecast(self, scaler, test, start_frame, num_frame, lookback):
        predictions = []
        for i in range(lookback, 0, -1):
            predictions.append([test[start_frame - i][0], test[start_frame - i][1]])

        for i in range(num_frame):
            prv_sample = np.empty((lookback, 2))
            for j in range(lookback):
                prv_sample[j] = predictions[j - lookback]
            sample = compute_sample(prv_sample)
            sample_scaled = scale_data(scaler, np.hstack((sample, np.array([[0, 0]]))))
            X = sample_scaled[:, : sample.shape[1]]
            yhat = self.model.predict(X)
            yhat_inv = invert_scale(scaler, np.hstack((X, yhat)))[:, -2: ]
            predictions.append([yhat_inv[0][0], yhat_inv[0][1]])

        return np.array(predictions).reshape((len(predictions), 2))


def compute_sample(data):
    sample = [data[-1][0], data[-1][1]]
    for i in range(len(data) - 1, 0, -1):
        x1, y1 = data[i]
        x0, y0 = data[i - 1]
        sample.append(x1 - x0)
        sample.append(y1 - y0)
    return np.array([sample])


def scale(data):
    array = data.reshape(data.shape[0], data.shape[1])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(array)
    data_scaled = scaler.transform(array)
    return scaler, data_scaled


def scale_data(scaler, data):
    array = data.reshape(data.shape[0], data.shape[1])
    return scaler.transform(array)


def invert_scale(scaler, data):
    array = data.reshape(data.shape[0], data.shape[1])
    inverted = scaler.inverse_transform(array)
    return inverted


def preprocess(data, lookback=3):
    x = np.empty((len(data), lookback * 2), dtype=np.float32)  # x3, y3, vx3, vy3, vx2, vy2, vx1, vy1
    y = np.empty((len(data), 2), dtype=np.float32)  # x4, y4 (current)
    for i in range(lookback, len(data) - 1):
        temp = [data[i][0], data[i][1], ]
        y[i] = data[i + 1]
        for j in range(lookback-1):
            x2, y2 = data[i - j]
            x1, y1 = data[i - j - 1]
            temp.append(x2 - x1)
            temp.append(y2 - y1)
        x[i] = temp
    return x[lookback: -1], y[lookback: -1]


def save_model(model, scaler, filename):
    print 'saving model to ' + filename + '...'
    pickle.dump((model, scaler), open(filename, 'wb'))


def load_model(filename):
    model, scaler = pickle.load(open(filename, 'rb'))
    return model, scaler


def load_parameters():
    return open("parameters.txt", 'r').readlines()[-1].split(',')