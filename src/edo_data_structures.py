import numpy as np
import torch
import copy
from torch.utils.data import Dataset
from src.EDOs import EDO
from typing import Iterable, Dict


class EdoData():
    """Class permettant de générer des données à partir de différentes EDOs."""
    def __init__(
            self,
            edo: EDO,
            initial_conditions: Iterable[float],
            noise_level: float = 0.0,
            dt: float = 0.01,
            **params: Dict[str, float | tuple]
        ):
        """
        Initialise les paramètres pour la génération de données à partir d'une EDO.
        Args:
            edo (EDO): Instance de la classe EDO représentant l'équation différentielle.
            initial_conditions (Iterable[float]): Conditions initiales pour l'EDO.
            noise_level (float): Niveau de bruit à ajouter aux données générées.
            dt (float): Pas de temps pour la simulation.
            **params: Paramètres spécifiques à l'EDO. Chaque paramètre peut être une valeur fixe ou un tuple (min, max) pour un échantillonnage aléatoire.
        """
        self.edo = edo
        self.params = params
        self.init_conditions = torch.tensor(initial_conditions, dtype=torch.float32)
        self.noise = noise_level
        self.dt = dt

    def evaluate(self, n, t_min, t_max) -> torch.Tensor:
        """
        Génère n séquences de données de l'EDO spécifiée.
        """
        data = np.random.normal(0, self.noise, (n, int((t_max-t_min)/self.dt), len(self.init_conditions)))

        for i in range(n):
            params = copy.deepcopy(self.params)
            for key, value in self.params.items():
                if isinstance(value, tuple):
                    params[key] = np.random.uniform(value[0], value[1])

            edo = self.edo(self.init_conditions.numpy(), **params)
            edo.resoudre_EDO(t_min, t_max, nombre_t=int((t_max - t_min) / self.dt))
            data[i, :, :] += edo.x_points.T

        return torch.tensor(data, dtype=torch.float32)


class EdoDataset(Dataset):
    """Dataset PyTorch pour les données générées à partir d'une EDO."""
    def __init__(self, edo_data: EdoData, n_samples: int, t_min: float, t_max: float):
        """
        Initialise le dataset avec les données générées.
        Args:
            edo_data (EdoData): Instance de la classe EdoData pour générer les données.
            n_samples (int): Nombre d'échantillons à générer.
            t_min (float): Temps minimum pour la simulation.
            t_max (float): Temps maximum pour la simulation.
        """
        self.data = edo_data.evaluate(n_samples, t_min, t_max)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx, :-1, :], self.data[idx, -1, :]
