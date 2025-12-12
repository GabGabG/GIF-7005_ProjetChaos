"""Simplified LSTM training on a single Lorenz trajectory with sliding windows."""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from src.EDOs import Lorenz
from src.edo_data_structures import EdoData, EdoDataset
from src.models.lstm import LSTM


torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hyperparamètres
    learning_rate = 0.0001
    nb_epochs = 2000
    batch_size = 64
    sequence_length = 100
    hidden_size = 128
    num_layers = 2

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
        t_max=120.0,
        sequence_length=sequence_length,
        stride=1
    )

    # Split train/val sur les fenêtres
    train_size = int(0.8 * len(dataset))
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

    save_path = "results/training/lstm/"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}{save_name}_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("LSTM Training (single trajectory) Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_name}_loss_curve.png", dpi=150)
    plt.close()

    # Évaluation autonome sur la suite de la trajectoire
    raw = data_generator.evaluate(0.0, 120.0)[0].cpu().numpy()  # (T, 3)
    split_idx = int(0.8 * len(raw))
    x_test = raw[split_idx:]
    start = raw[split_idx - 1]
    preds = model.predict(start, len(x_test))

    # Visualisation des prédictions
    rmse = np.sqrt(np.mean((preds - x_test) ** 2))
    rmse_dim = np.sqrt(np.mean((preds - x_test) ** 2, axis=0))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    dims = ['X', 'Y', 'Z']
    n_plot = min(2000, len(x_test))
    for i, (ax, dim) in enumerate(zip(axes, dims)):
        ax.plot(x_test[:n_plot, i], label='True', linewidth=2, alpha=0.8)
        ax.plot(preds[:n_plot, i], label='Pred', linewidth=1.5, alpha=0.7)
        ax.set_ylabel(dim)
        ax.set_xlabel('Steps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{dim} - RMSE: {rmse_dim[i]:.6f}')
    plt.tight_layout()
    plt.savefig(f"{save_path}{save_name}_predictions.png", dpi=150)
    plt.close()
    