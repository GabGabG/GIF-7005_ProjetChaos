import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from typing import Iterable
import warnings


class EDO:

    def __init__(self, conditions_initiales: Iterable[float]) -> None:
        self.conditions_initiales = conditions_initiales
        self.x_points = None
        self.t_points = None

    def equations_differentielles(self, t: float, x: Iterable[float]) -> np.ndarray:
        raise NotImplementedError("À implémenter dans les sous-classes.")

    def resoudre_EDO(self, t_min: float, t_max: float, nombre_t: int = None, ) -> None:
        t_eval = None
        if nombre_t is not None:
            t_eval = np.linspace(t_min, t_max, nombre_t)
        sol = solve_ivp(self.equations_differentielles, (t_min, t_max), self.conditions_initiales, t_eval=t_eval)
        self.x_points = sol.y
        self.t_points = sol.t
        status = sol.status
        if status == -1:
            warnings.warn(f"Intégration échouée:\n{sol.message}", RuntimeWarning)

    def enregistrer_solutions_csv(self, nom_fichier: str, nom_colonnes: Iterable[str]) -> pd.DataFrame:
        if self.t_points is None:
            raise ValueError("Solutions indisponibles. Assurez-vous d'appeler `resoudre_EDO`.")
        if not nom_fichier.lower().endswith(".csv"):
            nom_fichier += ".csv"
        data = self.x_points.T
        df = pd.DataFrame(data, self.t_points, nom_colonnes)
        df.index.names = ["Temps"]
        df.to_csv(nom_fichier)
        return df


class Lorenz(EDO):

    def __init__(self, conditions_initiales: Iterable[float], sigma: float = 10, rho: float = 28,
                 beta: float = 8 / 3) -> None:
        super(Lorenz, self).__init__(conditions_initiales)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def equations_differentielles(self, t: float, x: Iterable[float]) -> np.ndarray:
        # Équations indépendantes de t
        var_x, var_y, var_z = x
        var_x_update = self.sigma * (var_y - var_x)
        var_y_update = var_x * (self.rho - var_z) - var_y
        var_z_update = var_x * var_y - self.beta * var_z
        return np.array([var_x_update, var_y_update, var_z_update])
