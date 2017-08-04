import numpy as np

from commons import preprocess, GBM, save_model, scale


def build_model(n_node, n_estimator, lookback, model_fle):
    train_data = np.genfromtxt('training_data.txt', dtype=int, delimiter=",")
    train_sample, train_target = preprocess(train_data, lookback=lookback)
    sample_dim = train_sample.shape[1]
    scaler, scaled_train = scale(np.hstack((train_sample, train_target)))
    scaled_x = scaled_train[:, :sample_dim]
    scaled_y = scaled_train[:, sample_dim:]
    model = GBM(train_x=scaled_x, train_y=scaled_y, n_estimators=n_estimator, nodes=n_node)
    model.fit_gbm()
    save_model(model=model, scaler=scaler, filename=model_fle)

