""""""
import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy


class BaseModel(nn.Module):
    """Modèle de base pour les réseaux de neurones."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("La méthode forward doit être implémentée dans les sous-classes.")
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle à l'emplacement spécifié."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Charge le modèle depuis l'emplacement spécifié."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)
        return self
    
    def training_loop(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            n_epochs: int,
            lr: float,
    ):
        """
        Boucle d'entraînement du modèle.
        Args:
            train_loader (DataLoader): DataLoader pour les données d'entraînement.
            val_loader (DataLoader): DataLoader pour les données de validation.
            n_epochs (int): Nombre d'époques d'entraînement.
            lr (float): Taux d'apprentissage.
        Returns:
            train_losses (list): Liste des pertes d'entraînement par époque.
            val_losses (list): Liste des pertes de validation par époque.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(n_epochs), desc="Entraînement"):
            self.train()
            total_loss = 0.0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)

            avg_train_loss = total_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)

            self.eval()
            val_total_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self(inputs)
                    val_loss = criterion(outputs, targets)
                    val_total_loss += val_loss.item() * inputs.size(0)

            avg_val_loss = val_total_loss / len(val_loader.dataset)
            val_losses.append(avg_val_loss)

            # Sauvegarder le modèle si la perte de validation s'améliore
            if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
                best_state =deepcopy(self.state_dict())

        # Charger le meilleur modèle à la fin de l'entraînement
        self.load_state_dict(best_state)

        return train_losses, val_losses
    
    def predict(
            self,
            data_loader: torch.utils.data.DataLoader,
            return_cpu: bool = True,
        ) -> torch.Tensor:
        """
        Effectue une prédiction avec le modèle.
        Args:
            data_loader (DataLoader): DataLoader pour les données de test.
            return_cpu: Si True, retourne les prédictions sur CPU.
        Returns:
            torch.Tensor: Sortie prédite par le modèle.
        """
        self.eval()
        outputs = []
        with torch.no_grad():
            for test_data, _ in data_loader:
                test_data = test_data.to(self.device)
                predictions = self(test_data)
                outputs.append(predictions if not return_cpu else predictions.cpu())
        return torch.cat(outputs, dim=0)

    def predict_n_steps(
        self,
        initial_state: torch.Tensor,
        n_steps: int,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Prédit auto-récursivement sur n pas de temps à partir d'un seul tenseur d'état initial.
        Args:
            initial_state: Tenseur d'entrée initial.
            n_steps: Nombre de pas de temps à prédire.
            return_cpu: Si True, retourne les prédictions sur CPU.
        Returns:
            Tenseur de forme `(n_steps, features)` contenant les états prédits.
        """
        self.eval()
        with torch.no_grad():
            x = initial_state.to(self.device)
            predictions = []
            for _ in range(n_steps):
                x = self(x)
                predictions.append(x if not return_cpu else x.cpu())

        return torch.cat(predictions, dim=0)
