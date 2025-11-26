from src.EDOs import Lorenz
from src.Reservoir import ESN, GridOptim
from src.ChaosCharacterization import TentMap
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tsteps = np.arange(0, 150, 0.02)
    L = Lorenz((1, 0, 0))
    L.resoudre_EDO(0, tsteps[-1], t_eval=tsteps)
    x_vals = L.x_points.T  # On veut shape (t, x)
    n_train = 5000
    x_train = x_vals[:n_train]
    y_train = x_vals[1:n_train + 1]
    x_test = x_vals[n_train:]
    y_test = x_vals[n_train:]
    ts = y_test[..., -1]

    t_vals = L.t_points
    t_test = t_vals[n_train:] - t_vals[n_train]
    ridge_options = np.append([0], np.logspace(-8, 0, 9))

    # ridge = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    # n = [300, 500, 700]
    # gopt = GridOptim({"ridge": ridge, "n_units": n}, n_input=3, wanted_sr=1.2, seed_W=42, seed_W_in=43)
    # best = gopt.optimize(x_train, y_train, x_test[0], y_test, ESN.prediction_error)
    # print(best)
    best_ridge = 0.0001
    print(best_ridge)
    res = ESN(500, 3, 1.2, ridge=best_ridge, seed_W=42, seed_W_in=43)
    res.fit(x_train, y_train, skip=800)
    n_predict = len(tsteps) - n_train
    print(n_predict)
    # plt.plot(x_vals)
    # plt.show()
    preds = res.autonomous_prediction(x_test[0], n_predict)
    fig, axes = plt.subplots(3, 1)
    labels = [r"$x$", r"$y$", r"$z$"]
    for i in range(3):
        axes[i].plot(t_test[:], y_test[:, i], label="Vraies")
        axes[i].plot(t_test[:], preds[:, i], label="Prédictions")
        axes[i].set_xlabel("Temps [-]")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    plt.show()

    fig, ax = plt.subplots()
    tm_base = TentMap(y_test, -1)
    tm_base.compute_tent_map(True)
    tm_base.display_tent_map(ax, label="Vraies")
    tm_preds = TentMap(preds, -1)
    tm_preds.compute_tent_map(True)
    tm_preds.display_tent_map(ax, color="red", marker="x", label="Prédictions")
    ax.set_xlabel(r"$z_n$")
    x = min(tm_base.tent_map[0])
    ax.axline((x, x), slope=1, color="black", ls="--", label=r"$y = x$")
    ax.set_ylabel(r"$z_{n+1}$")
    ax.legend()
    plt.show()
