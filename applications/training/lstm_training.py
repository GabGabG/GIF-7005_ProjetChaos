import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from src.EDOs import Lorenz
from src.edo_data_structures import EdoData, EdoDataset, normalisation, denormalisation
from src.models.lstm import LSTM
from src.ChaosCharacterization import TentMap


torch.manual_seed(42)
np.random.seed(42)

plt.rcParams.update({'font.size': 22})

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparamètres
    learning_rate = 0.001
    nb_epochs = 1000
    batch_size = 128
    sequence_length = 1
    hidden_size = 32
    num_layers = 1

    save_name = f"lstm_lr{learning_rate}_epochs{nb_epochs}_L{sequence_length}_H{hidden_size}"

    data_generator = EdoData(
        edo=Lorenz,
        initial_conditions=[1.0, 0.0, 0.0],
        noise_level=0.0,
        dt=0.02,
        sigma=10,
        rho=28,
        beta=8/3
    )

    dataset = EdoDataset(
        edo_data=data_generator,
        t_min=0.0,
        t_max=100.0,
        sequence_length=sequence_length,
        stride=1,
        normalize=True
    )

    # Split train/val sur les fenêtres
    train_size = 4000
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"  - Fenêtres totales: {len(dataset)} (train: {len(train_data)}, val: {len(val_data)})")

    # Entraînement du LSTM en next-step
    model = LSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers, output_size=3).to(device)
    train_losses, val_losses = model.training_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=nb_epochs,
        lr=learning_rate
    )

    save_path = "results/lstm/"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}{save_name}_model.pth")
    print(f"Model sauvegarder : {save_path}{save_name}_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(train_losses), label="Train Loss", linewidth=2)
    plt.plot(np.log10(val_losses), label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("log(MSE Loss)")
    plt.legend()
    plt.title("LSTM Training Loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_name}_loss_curve.png", dpi=150)
    plt.show()

    # Évaluation autonome sur la suite de la trajectoire
    raw = data_generator.evaluate(0.0, 200.0)[0].cpu()  # (T, 3)
    split_idx = 5000
    
    # Utiliser les paramètres de normalisation du dataset d'entraînement
    if hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
        mean, std = dataset.mean, dataset.std
        raw_normalized = (raw - mean) / std
    else:
        mean, std = None, None
        raw_normalized = raw
    
    x_test = raw_normalized[split_idx:]
    # Utiliser une séquence initiale de longueur sequence_length
    start_sequence = raw_normalized[split_idx - sequence_length:split_idx]
    preds = model.predict(start_sequence.numpy(), len(x_test))

    # Dénormaliser pour obtenir les vraies valeurs
    if mean is not None and std is not None:
        x_test = denormalisation(torch.tensor(x_test), mean, std).numpy()
        preds = denormalisation(torch.tensor(preds), mean, std).numpy()
    else:
        x_test = x_test.numpy()
        preds = preds.astype(np.float32)

    n_plot = 1000
    t_plot = np.linspace(0, 20, n_plot)
    rmse = np.sqrt(np.mean((preds - x_test)[:n_plot] ** 2))
    rmse_dim = np.sqrt(np.mean((preds - x_test)[:n_plot] ** 2, axis=0))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    dims = ['X', 'Y', 'Z']

    for i, (ax, dim) in enumerate(zip(axes, dims)):
        ax.plot(t_plot, x_test[:n_plot, i], label='True', linewidth=2, alpha=0.8)
        ax.plot(t_plot, preds[:n_plot, i], label='Pred', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(dim)
        if i == 2:
            ax.set_xlabel('Temps $t$')
        if i == 0:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{dim} - RMSE: {rmse_dim[i]:.6f}')
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_name}_predictions.png", dpi=150)
    plt.show()
    print(f"Prediction RMSE: {rmse:.6f}")

    # Visulalisation des résidus
    diff_abs = np.abs(preds - x_test)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, (ax, dim) in enumerate(zip(axes, dims)):
        ax.plot(t_plot, diff_abs[:n_plot, i], label='Diff. absolue', color='orange', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(dim)
        if i == 2:
            ax.set_xlabel('Temps $t$')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{dim} - Diff. absolue')
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_name}_residuals.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))
    tm_base = TentMap(x_test, -1)
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
    fig.savefig("tent_map_lstm.png", dpi=300)
    plt.show()
    