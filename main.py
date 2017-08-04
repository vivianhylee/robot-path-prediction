import os
import numpy as np
from commons import preprocess, GBM, load_model, scale, invert_scale


def run():
    ipt_dir = 'inputs'
    out_dir = 'output'

    filenames = os.listdir(ipt_dir)
    model_fle, lookback, n_node, n_estimator = 'model.plk', 20, 15, 900

    # Build model: uncomment if no trained model
    #build_model(n_node=n_node, n_estimator=n_estimator, lookback=lookback, model_fle=model_fle)
    model, scaler = load_model(model_fle)

    for fle in filenames:
        test_input = np.genfromtxt(os.path.join(ipt_dir, fle), dtype=int, delimiter=",")
        model, scaler = load_model(model_fle)
        predictions = model.forecast(scaler=scaler, test=test_input, start_frame=1740, num_frame=60, lookback=int(lookback))[int(lookback):]

        with open(os.path.join(out_dir, 'res_' + fle), 'w') as f:
            for val in predictions:
                x, y = val
                print >> f, '%s,%s' % (int(x), int(y))


if __name__ == '__main__':

    run()

