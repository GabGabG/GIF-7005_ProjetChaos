from src.EDOs import Lorenz
from src.Reservoir import ESN
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tsteps = np.arange(0, 120, 0.02)
    L = Lorenz((1, 0, 0))
    L.resoudre_EDO(0, tsteps[-1], t_eval=tsteps)
    x_vals = L.x_points.T  # On veut shape (t, x)
    n_train = 5000
    x_train = x_vals[:n_train]
    y_train = x_vals[1:n_train + 1]
    x_test = x_vals[n_train:]
    y_test = x_vals[n_train:]
    t_vals = L.t_points
    t_test = t_vals[n_train:] - t_vals[n_train]
    res = ESN(300, 3, 1.2, ridge=1e-5)
    res.fit(x_train, y_train, skip=800)
    n_predict = len(tsteps) - n_train
    # plt.plot(x_vals)
    # plt.show()
    preds = res.autonomous_prediction(x_test[0], n_predict)
    fig, axes = plt.subplots(3, 1)
    labels = [r"$x$", r"$y$", r"$z$"]
    for i in range(3):
        axes[i].plot(t_test, y_test[..., i], label="Vraies")
        axes[i].plot(t_test, preds[..., i], label="Prédictions")
        axes[i].set_xlabel("Temps [-]")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    plt.show()
    error = res.prediction_error(preds, y_test, False)
    print(error, res.prediction_error(preds, y_test))
