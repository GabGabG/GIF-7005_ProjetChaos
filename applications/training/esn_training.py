from src.EDOs import Lorenz
from src.models.Reservoir import ESN
from src.ChaosCharacterization import TentMap
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

if __name__ == '__main__':
    tsteps = np.arange(0, 200, 0.02)
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

    res = ESN(n_units=300, n_input=3, wanted_sr=1.2, seed_W=42, seed_W_in=42, ridge=1e-3)
    res.fit(x_train, y_train, skip=800)
    n_predict = len(tsteps) - n_train

    n_plot = 1000
    preds = res.autonomous_prediction(x_test[0], n_predict)
    rmse = np.sqrt(np.mean((preds - y_test)[:n_plot] ** 2))
    rmse_dim = np.sqrt(np.mean((preds - y_test)[:n_plot] ** 2, axis=0))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dims = ['$x$', '$y$', '$z$']
    for i, (ax, dim) in enumerate(zip(axes, dims)):
        ax.plot(t_test[:n_plot], x_test[:n_plot, i], label='True', linewidth=2, alpha=0.8)
        ax.plot(t_test[:n_plot], preds[:n_plot, i], label='Pred', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(dim)
        if i == 2:
            ax.set_xlabel('Temps $t$')
        if i == 0:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{dim} - RMSE: {rmse_dim[i]:.6f}')
    plt.tight_layout()
    plt.savefig(f"results/esn/ESN_lorenz_predictions.png", dpi=150)
    plt.show()
    print(f"Prediction RMSE: {rmse:.6f}")

    diff_abs = np.abs(preds - x_test)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, (ax, dim) in enumerate(zip(axes, dims)):
        ax.plot(t_test[:n_plot], diff_abs[:n_plot, i], label='Diff. absolue', color='orange', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(dim)
        if i == 2:
            ax.set_xlabel('Temps $t$')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{dim} - Diff. absolue')
        # ax.set_yscale("symlog", base=10)
    plt.tight_layout()
    plt.savefig(f"results/esn/ESN_lorenz_residuals.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))
    tm_base = TentMap(y_test, -1)
    tm_base.compute_tent_map(True)
    tm_base.display_tent_map(ax, label="True")
    tm_preds = TentMap(preds, -1)
    tm_preds.compute_tent_map(True)
    tm_preds.display_tent_map(ax, color="red", marker="x", label="Pred")
    ax.set_xlabel(r"$z_n$")
    x = min(tm_base.tent_map[0])
    ax.axline((x, x), slope=1, color="black", ls="--", label=r"$y = x$")
    ax.set_ylabel(r"$z_{n+1}$")
    ax.legend()
    plt.tight_layout()
    fig.savefig("tent_map_lorenz.png", dpi=150)
    plt.show()
