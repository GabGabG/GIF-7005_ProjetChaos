import numpy as np
import torch
import copy
from torch.utils.data import Dataset
from src.EDOs import EDO
from typing import Iterable, Dict


class EdoData():
    """Générateur de données basé sur UNE seule trajectoire d'une EDO.

    Simplifie la génération: on produit une unique série temporelle continue
    (ex. Lorenz) sur l'intervalle [t_min, t_max] avec pas `dt` et on l'utilise
    ensuite pour construire des fenêtres d'entraînement.
    """
    def __init__(
            self,
            edo: EDO,
            initial_conditions: Iterable[float],
            noise_level: float = 0.0,
            dt: float = 0.01,
            **params: Dict[str, float]
        ):
        """
        Initialise les paramètres pour la génération de données à partir d'une EDO.
        Args:
            edo (EDO): Instance de la classe EDO représentant l'équation différentielle.
            initial_conditions (Iterable[float]): Conditions initiales pour l'EDO.
            noise_level (float): Niveau de bruit à ajouter aux données générées.
            dt (float): Pas de temps pour la simulation.
            **params: Paramètres spécifiques à l'EDO.
        """
        self.edo = edo
        self.params = params
        self.init_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
        self.noise = noise_level
        self.dt = dt

    def evaluate(self, t_min: float, t_max: float) -> torch.Tensor:
        """Génère et retourne UNE trajectoire au format (1, T, F).

        Args:
            t_min: début de l'intervalle temporel
            t_max: fin de l'intervalle temporel
        Returns:
            Tensor de shape (1, T, F) contenant la trajectoire unique.
        """
        T = int((t_max - t_min) / self.dt)
        edo = self.edo(self.init_conditions.numpy(), **self.params)
        edo.resoudre_EDO(t_min, t_max, nombre_t=T)
        trajectory = torch.tensor(edo.x_points.T, dtype=torch.float32)  # (T, F)
        if self.noise > 0.0:
            trajectory = trajectory + torch.randn_like(trajectory) * self.noise
        return trajectory.unsqueeze(0)


class EdoDataset(Dataset):
    """Dataset PyTorch basé sur une trajectoire, avec fenêtres glissantes.

    - Construit des fenêtres de longueur L+1 pour entraînement next-step:
        inputs = L premières valeurs, target = valeur suivante.
    """
    def __init__(
            self,
            edo_data: EdoData,
            t_min: float,
            t_max: float,
            sequence_length: int = 50,
            stride: int = 1,
            normalize: bool = False
        ):
        self.sequence_length = sequence_length
        self.stride = stride

        # Génération de la trajectoire
        self.raw_data = edo_data.evaluate(t_min, t_max)  # (1, T, F)
        if normalize:
            self.raw_data, self.mean, self.std = normalisation(self.raw_data)

        # Construction des fenêtres
        self.data = self._create_sequences()

    def _create_sequences(self) -> torch.Tensor:
        """Crée des fenêtres glissantes de longueur L+1 pour next-step.

        Retourne un tenseur de shape (N, L+1, F). Le `__getitem__` découpe
        ensuite en `(inputs=L, F)` et `target=(F,)`.
        """
        L = self.sequence_length
        sequences = []
        sample = self.raw_data[0]  # (T, F) unique trajectoire
        T = sample.shape[0]
        for start_idx in range(0, T - (L + 1), self.stride):
            seq = sample[start_idx:start_idx + (L + 1)]  # (L+1, F)
            sequences.append(seq)
        return torch.stack(sequences) if sequences else sample.unsqueeze(0)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (inputs, target) pour next-step.

        - inputs: (L, F) → les L premières valeurs de la fenêtre
        - target: (F,)   → la valeur immédiatement suivante (pas L+1)
        """
        window = self.data[idx]
        return window[:-1, :], window[-1, :]
    
    def denormalize(self, normalized_data: torch.Tensor) -> torch.Tensor:
        """Dénormalise les données en utilisant la moyenne et l'écart-type stockés.

        Args:
            normalized_data: Tensor de shape (N, L, F) ou (T, F)
        Returns:
            données dénormalisées
        """
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            return denormalisation(normalized_data, self.mean, self.std)

def normalisation(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalise les données par feature (moyenne 0, écart-type 1).

    Args:
        data: Tensor de shape (N, L, F) ou (T, F)
    Returns:
        normalized_data: données normalisées
        mean: moyenne par feature
        std: écart-type par feature
    """
    mean = data.mean(dim=(0, 1), keepdim=True)
    std = data.std(dim=(0, 1), keepdim=True) + 1e-8  # éviter division par zéro
    normalized_data = (data - mean) / std
    return normalized_data, mean.squeeze(), std.squeeze()

def denormalisation(normalized_data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse la normalisation des données.

    Args:
        normalized_data: Tensor de shape (N, L, F) ou (T, F)
        mean: moyenne par feature
        std: écart-type par feature
    Returns:
        données dénormalisées
    """
    return normalized_data * std + mean
