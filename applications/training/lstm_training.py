""""""
import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from src import *


torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    nb_epochs = 50
    save_name = f"lstm_lr{learning_rate}_epochs{nb_epochs}"

    data_generator = EdoData(
        edo=Lorenz,
        initial_conditions=[1.0, 1.0, 1.0],
        noise_level=0.0,
        dt=0.01,
        sigma=(5, 15),
        rho=(20, 35),
        beta=(2, 3)
    )
    dataset = EdoDataset(data_generator, n_samples=10000, t_min=0.0, t_max=10.0)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model = LSTM(input_size=3, hidden_size=64, num_layers=2, output_size=3).to(device)
    train_losses, val_losses = model.training_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=nb_epochs,
        lr=learning_rate
    )

    save_path = "results/training/lstm/"
    os.makedirs(save_path, exist_ok=True)

    model.save(f"{save_path}{save_name}_model.pth")

    # Courbe des pertes
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LSTM Training and Validation Loss")
    plt.savefig(f"{save_path}{save_name}_loss_curve.png")
    plt.clf()

    # Test du modèle
    test_dataset = EdoDataset(data_generator, n_samples=256, t_min=0, t_max=10.0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = model.predict(test_loader)
    actuals = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)
    mse = torch.mean((predictions - actuals) ** 2).item()
    print(f"Test MSE: {mse}")
    